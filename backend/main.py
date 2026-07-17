# Backend for 憶起 — M55M1 smart photo frame (Day 2-3) + Flutter app (App stage).
#
# Endpoints:
#   -- App: trips (App Stage 1, SQLite via db.py) --
#   POST   /trips/start             -> create a trip, returns trip_id
#   POST   /trips/{id}/points       -> batch-append GPS points
#   POST   /trips/{id}/end          -> close the trip, compute distance
#   GET    /trips                   -> trip list (summaries, newest first)
#   GET    /trips/{id}              -> trip detail (points + photos + story)
#   DELETE /trips/{id}              -> delete trip (+ points/photos/files)
#
#   -- M55M1 frame (Day 2-3, unchanged) --
#   GET  /ping                      -> liveness check                       (Day 2)
#   POST /story                     -> canned story text                    (Day 2, kept for compat)
#   POST /generate {face_id}        -> LLM story (zh+en) + audio_id;        (Day 3 Task 1)
#                                      TTS starts in the background
#   GET  /audio/{audio_id}          -> 16kHz/16bit/mono WAV; blocks until   (Day 3 Task 2)
#                                      the TTS job finishes (max ~90 s)
#   GET  /audio/{audio_id}/status   -> {"status": pending|ready|error, "size": N}
#   GET  /textimg/{audio_id}        -> story strip for the LCD (TIM4 4-bpp,     (Day 3 Task 3)
#                                      see textimg.py; ready as soon as
#                                      /generate returns)
#   GET  /textimg/{audio_id}/preview-> same strip as PNG (browser check)
#
# Environment (LLM routing details at the LLM_MODEL block below):
#   LLM_MODEL       model name, default gemini-3-flash (provider auto-prefixed)
#   GEMINI_API_KEY  key when calling Google AI Studio directly
#   LLM_API_BASE +  when going through an OpenAI-compatible gateway
#   OPENAI_API_KEY  (e.g. LiteLLM proxy) that serves the model
#   LLM_MOCK=1      skip the LLM and use a canned story (keyless pipeline testing)
#
# Run:  uvicorn main:app --host 0.0.0.0 --port 8000
import asyncio
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import litellm  # heavy import (~10 s); pay it at startup, not on first /generate
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

import db
from db import Frame, GpsPoint, Photo, Trip, TripSpot
from textimg import render_story_tim4


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_db()
    # 開發期種子：還沒有任何相框就建一台 demo 相框（配對碼 123456）。
    # 未來 M55M1 會在 LCD 顯示自己的配對碼、由註冊流程建檔。
    async with db.SessionLocal() as session:
        has_frame = (await session.execute(select(Frame.id).limit(1))).scalar()
        if has_frame is None:
            session.add(Frame(pair_code="123456", name="客廳的相框"))
            await session.commit()
            print("[frames] seeded demo frame (pair_code=123456)")
    yield


app = FastAPI(title="憶起 backend", lifespan=lifespan)

# Flutter Web（flutter run -d chrome）從 localhost:<隨機port> 打過來是跨來源，
# 瀏覽器要求後端帶 CORS header 才放行；App 真機/模擬器不走瀏覽器、不受影響。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發用；正式部署要收斂
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_DIR = Path(__file__).parent / "photo_store"
PHOTO_DIR.mkdir(exist_ok=True)

# DB 存 naive-UTC；給使用者看的日期（預設標題、遊記 prompt）用台北時間。
TAIPEI_TZ = timezone(timedelta(hours=8))

AUDIO_DIR = Path(__file__).parent / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)

TTS_VOICE = "zh-TW-HsiaoChenNeural"

# --- LLM routing ---------------------------------------------------------
# litellm needs a provider prefix to know which protocol to speak; a bare
# "gemini-3-flash" raises "LLM Provider NOT provided" (verified). The model
# name itself is passed through to the provider untouched.
#   LLM_MODEL     model name, bare or prefixed          (default gemini-3-flash)
#   LLM_API_BASE  OpenAI-compatible gateway URL (e.g. a LiteLLM proxy);
#                 setting it makes bare model names go openai-protocol
#   LLM_API_KEY   explicit key for that gateway (else litellm reads the
#                 provider's usual env var)
LLM_API_BASE = (
    os.environ.get("LLM_API_BASE")
    or os.environ.get("OPENAI_API_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://litellm.netdb.csie.ncku.edu.tw"  # NCKU NetDB LiteLLM proxy
).rstrip("/")

# Gemini 3 "thinks" by default: measured 23.6 s vs 10.5 s with low effort on
# the same story prompt. Set LLM_REASONING_EFFORT= (empty) to disable the param
# (needed for models that reject it, e.g. gpt-4o-mini).
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")

_raw_model = os.environ.get("LLM_MODEL", "gemini-3-flash")
if "/" in _raw_model:
    LLM_MODEL = _raw_model                    # caller chose the provider
elif LLM_API_BASE:
    LLM_MODEL = f"openai/{_raw_model}"        # OpenAI-compatible gateway
elif _raw_model.startswith("gemini"):
    LLM_MODEL = f"gemini/{_raw_model}"        # Google AI Studio (GEMINI_API_KEY)
else:
    LLM_MODEL = _raw_model


def required_api_keys(model: str) -> list[str]:
    """Env vars (any one suffices) litellm reads for this model's provider."""
    if model.startswith("gemini/"):
        return ["GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"]
    return ["OPENAI_API_KEY", "LLM_API_KEY"]

# audio_id -> {"status": "pending"|"ready"|"error", "wav": Path, "error": str}
AUDIO_JOBS: dict[str, dict] = {}

FAKE_TRIPS = {
    1: {
        "name": "爸爸",
        "name_en": "Dad",
        "place": "日月潭",
        "place_en": "Sun Moon Lake",
        "date": "2026年6月14日",
        "highlights": ["清晨搭船遊湖看日出", "向山自行車道騎車", "伊達邵老街吃小米麻糬"],
    },
    2: {
        "name": "媽媽",
        "name_en": "Mom",
        "place": "九份",
        "place_en": "Jiufen",
        "date": "2026年5月3日",
        "highlights": ["阿妹茶樓喝茶看海", "黃昏的豎崎路紅燈籠", "買了手工芋圓當伴手禮"],
    },
    3: {
        "name": "全家",
        "name_en": "the whole family",
        "place": "墾丁",
        "place_en": "Kenting",
        "date": "2026年4月4日",
        "highlights": ["白沙灣玩水堆沙堡", "鵝鑾鼻燈塔野餐", "夜晚在墾丁大街吃烤魷魚"],
    },
}


@app.get("/ping")
def ping():
    return {"status": "ok", "timestamp": time.time()}


class FaceEvent(BaseModel):
    face_id: int
    timestamp: float


@app.post("/story")
def get_story(event: FaceEvent):
    stories = {
        1: "你好，這是你上週去日月潭的回憶",
        2: "歡迎回來，來看看上個月的九份之旅",
    }
    return {
        "face_id": event.face_id,
        "story": stories.get(event.face_id, "還沒有你的旅遊回憶喔"),
    }


# ---------------------------------------------------------------- Task 1: LLM

class GenerateRequest(BaseModel):
    face_id: int


def build_prompt(trip: dict) -> str:
    return (
        "你是智慧相框的溫馨旁白。根據以下旅遊資料，寫一段給家人聽的故事旁白。\n"
        f"人物：{trip['name']}\n"
        f"地點：{trip['place']}\n"
        f"日期：{trip['date']}\n"
        f"亮點：{'、'.join(trip['highlights'])}\n\n"
        "要求：\n"
        "1. story_zh：繁體中文，40到55個字（硬上限60字，超過會被截斷），"
        "語氣溫暖親切，像在對家人說話、喚起美好回憶。"
        "不要引號、不要標題，直接是旁白內容。\n"
        "2. story_en：上述旁白的英文版，一到兩句、簡潔自然。\n"
        '請只輸出 JSON：{"story_zh": "...", "story_en": "..."}'
    )


def call_llm(trip: dict) -> tuple[str, str]:
    """Returns (story_zh, story_en). Raises HTTPException on failure."""
    if os.environ.get("LLM_MOCK") == "1":
        return (
            f"還記得{trip['date']}嗎？{trip['name']}在{trip['place']}留下了好多笑聲，"
            f"{trip['highlights'][0]}的畫面到現在都還暖暖的呢。",
            f"Remember {trip['place_en']}? {trip['name_en'].capitalize()} made such warm memories there.",
        )

    key_vars = required_api_keys(LLM_MODEL)
    if not any(os.environ.get(k) for k in key_vars):
        raise HTTPException(
            status_code=500,
            detail=f"none of {key_vars} is set for model {LLM_MODEL} "
            "(or use LLM_MOCK=1 for pipeline testing)",
        )

    extra = {}
    if LLM_API_BASE:
        extra["api_base"] = LLM_API_BASE
    if os.environ.get("LLM_API_KEY"):
        extra["api_key"] = os.environ["LLM_API_KEY"]
    if LLM_REASONING_EFFORT:
        extra["reasoning_effort"] = LLM_REASONING_EFFORT
        # client-side litellm blocks non-standard params on the openai route;
        # the proxy on the other end maps it to gemini thinking_level
        extra["allowed_openai_params"] = ["reasoning_effort"]

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": build_prompt(trip)}],
            response_format={"type": "json_object"},
            temperature=0.8,
            timeout=30,
            **extra,
        )
        data = json.loads(resp.choices[0].message.content)
        return data["story_zh"].strip(), data["story_en"].strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


# ---------------------------------------------------------------- Task 2: TTS

async def tts_to_wav(text: str, wav_path: Path):
    """edge-tts -> mp3, then ffmpeg -> WAV PCM 16 kHz / 16-bit / mono.

    -bitexact keeps the WAV header at the plain 44-byte RIFF layout (no LIST
    metadata chunk) so a naive header parser on the MCU still finds "data".
    Raises on failure."""
    import edge_tts

    mp3_path = wav_path.with_suffix(".mp3")
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(str(mp3_path))

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-bitexact",
            str(wav_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr.decode(errors='replace')[-400:]}")
    finally:
        mp3_path.unlink(missing_ok=True)


async def tts_job(audio_id: str, text: str):
    """Day-3 背景 TTS：完成後把 AUDIO_JOBS 狀態改 ready（/audio 端點在等它）。"""
    job = AUDIO_JOBS[audio_id]
    wav_path = AUDIO_DIR / f"{audio_id}.wav"
    try:
        await tts_to_wav(text, wav_path)
        job["status"] = "ready"
        print(f"[tts] {audio_id} ready: {wav_path.stat().st_size} bytes")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[tts] {audio_id} FAILED: {e}")


@app.post("/generate")
async def generate(req: GenerateRequest):
    trip = FAKE_TRIPS.get(req.face_id)
    if trip is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown face_id {req.face_id}, known: {sorted(FAKE_TRIPS)}",
        )

    t0 = time.time()
    story_zh, story_en = await asyncio.to_thread(call_llm, trip)
    llm_ms = int((time.time() - t0) * 1000)

    audio_id = uuid.uuid4().hex[:12]
    tim4 = render_story_tim4(
        story_zh,
        header=trip["place"],
        subheader=trip["date"],
        preview_png=AUDIO_DIR / f"{audio_id}.png",
    )
    AUDIO_JOBS[audio_id] = {
        "status": "pending",
        "wav": AUDIO_DIR / f"{audio_id}.wav",
        "error": "",
        "tim4": tim4,
    }
    asyncio.get_running_loop().create_task(tts_job(audio_id, story_zh))

    print(f"[generate] face_id={req.face_id} llm={llm_ms}ms audio_id={audio_id}")
    print(f"[generate] story_zh: {story_zh}")
    return {
        "face_id": req.face_id,
        "story_text": story_zh,
        "story_text_en": story_en,
        "audio_id": audio_id,
    }


@app.get("/textimg/{audio_id}")
def get_textimg(audio_id: str):
    """Story strip for the LCD: TIM4 4-bpp image (see textimg.py header)."""
    job = AUDIO_JOBS.get(audio_id)
    if job is None or "tim4" not in job:
        raise HTTPException(status_code=404, detail="unknown audio_id")
    return Response(content=job["tim4"], media_type="application/octet-stream")


@app.get("/textimg/{audio_id}/preview")
def get_textimg_preview(audio_id: str):
    """PNG preview of the strip in board colors (browser check only)."""
    png = AUDIO_DIR / f"{audio_id}.png"
    if not png.exists():
        raise HTTPException(status_code=404, detail="unknown audio_id")
    return FileResponse(png, media_type="image/png")


@app.get("/audio/{audio_id}/status")
def audio_status(audio_id: str):
    job = AUDIO_JOBS.get(audio_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown audio_id")
    size = job["wav"].stat().st_size if job["status"] == "ready" else 0
    return {"audio_id": audio_id, "status": job["status"], "size": size, "error": job["error"]}


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    job = AUDIO_JOBS.get(audio_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown audio_id")

    # TTS runs in the background after /generate; hold the request until done
    # so the MCU can GET immediately without a polling loop.
    deadline = time.time() + 90
    while job["status"] == "pending" and time.time() < deadline:
        await asyncio.sleep(0.25)

    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=f"TTS failed: {job['error']}")
    if job["status"] != "ready":
        raise HTTPException(status_code=504, detail="TTS still not ready after 90 s")

    return FileResponse(job["wav"], media_type="audio/wav", filename=f"{audio_id}.wav")


# ================================================================ App: trips
# Flutter app endpoints (App Stage 1). Storage in SQLite via db.py; datetimes
# arrive as ISO-8601 (any zone), are stored naive-UTC, and go back out as
# "...Z" strings (db.iso_z) so Flutter's DateTime.parse() round-trips cleanly.


class TripStartRequest(BaseModel):
    title: str = ""


class PointIn(BaseModel):
    lat: float
    lng: float
    timestamp: datetime


class PointsBatch(BaseModel):
    points: list[PointIn]


def trip_summary(trip: Trip, point_count: int, photo_count: int,
                 cover_photo_id: int | None) -> dict:
    return {
        "id": trip.id,
        "title": trip.title,
        "start_time": db.iso_z(trip.start_time),
        "end_time": db.iso_z(trip.end_time),
        "distance_m": round(trip.distance_m, 1),
        "has_story": bool(trip.story_text),
        "point_count": point_count,
        "photo_count": photo_count,
        "cover_photo_id": cover_photo_id,
        "created_at": db.iso_z(trip.created_at),
    }


async def get_trip_or_404(session, trip_id: int, *, with_children: bool = False) -> Trip:
    stmt = select(Trip).where(Trip.id == trip_id)
    if with_children:
        stmt = stmt.options(selectinload(Trip.points), selectinload(Trip.photos))
    trip = (await session.execute(stmt)).scalar_one_or_none()
    if trip is None:
        raise HTTPException(status_code=404, detail=f"unknown trip_id {trip_id}")
    return trip


@app.post("/trips/start")
async def start_trip(req: TripStartRequest):
    async with db.SessionLocal() as session:
        trip = Trip(title=req.title.strip(), start_time=db.utcnow())
        session.add(trip)
        await session.commit()
        print(f"[trips] started trip {trip.id}")
        return {"trip_id": trip.id, "start_time": db.iso_z(trip.start_time)}


@app.post("/trips/{trip_id}/points")
async def add_points(trip_id: int, batch: PointsBatch):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
        session.add_all(
            GpsPoint(
                trip_id=trip.id,
                lat=p.lat,
                lng=p.lng,
                timestamp=db.to_naive_utc(p.timestamp),
            )
            for p in batch.points
        )
        await session.commit()
        total = (
            await session.execute(
                select(GpsPoint.id).where(GpsPoint.trip_id == trip_id)
            )
        ).scalars().all()
        return {"added": len(batch.points), "total": len(total)}


@app.post("/trips/{trip_id}/end")
async def end_trip(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        if trip.end_time is None:
            trip.end_time = db.utcnow()
        trip.distance_m = db.path_distance_m([(p.lat, p.lng) for p in trip.points])
        if not trip.title and trip.start_time:
            local = trip.start_time.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
            trip.title = f"{local.month}月{local.day}日的旅程"
        await session.commit()
        print(f"[trips] ended trip {trip.id}: {len(trip.points)} pts, "
              f"{trip.distance_m:.0f} m")
        return trip_summary(
            trip, len(trip.points), len(trip.photos),
            trip.photos[0].id if trip.photos else None,
        )


@app.get("/trips")
async def list_trips():
    async with db.SessionLocal() as session:
        trips = (
            (await session.execute(
                select(Trip)
                .options(selectinload(Trip.points), selectinload(Trip.photos))
                .order_by(Trip.created_at.desc())
            )).scalars().all()
        )
        return {
            "trips": [
                trip_summary(
                    t, len(t.points), len(t.photos),
                    t.photos[0].id if t.photos else None,
                )
                for t in trips
            ]
        }


@app.get("/trips/{trip_id}")
async def trip_detail(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        detail = trip_summary(
            trip, len(trip.points), len(trip.photos),
            trip.photos[0].id if trip.photos else None,
        )
        detail["story_text"] = trip.story_text
        detail["points"] = [
            {"lat": p.lat, "lng": p.lng, "timestamp": db.iso_z(p.timestamp)}
            for p in trip.points
        ]
        detail["photos"] = [
            {
                "id": ph.id,
                "lat": ph.lat,
                "lng": ph.lng,
                "timestamp": db.iso_z(ph.timestamp),
                "url": f"/photos/{ph.id}",
            }
            for ph in trip.photos
        ]
        return detail


@app.delete("/trips/{trip_id}")
async def delete_trip(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        for ph in trip.photos:
            (PHOTO_DIR / ph.filename).unlink(missing_ok=True)
            for board in PHOTO_DIR.glob(f"board_{ph.id}_*.jpg"):
                board.unlink(missing_ok=True)
        for media in MEDIA_DIR.glob(f"t{trip_id}_*.*"):
            media.unlink(missing_ok=True)
        await session.execute(delete(TripSpot).where(TripSpot.trip_id == trip_id))
        await session.delete(trip)  # ORM cascade removes points + photos rows
        await session.commit()
        # SQLite 會重用 trip_id，殘留的景點快取會張冠李戴
        _SPOTS_POOL.pop(trip_id, None)
        _SPOTS_RESULT.pop(trip_id, None)
        print(f"[trips] deleted trip {trip_id}")
        return {"deleted": trip_id}


# ================================================================ App: photos
# multipart 上傳（App Stage 3）。檔案落地 photo_store/，中繼資料進 photos 表。

PHOTO_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
}


@app.post("/trips/{trip_id}/photos")
async def upload_photo(
    trip_id: int,
    file: UploadFile = File(...),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    timestamp: datetime | None = Form(None),
):
    async with db.SessionLocal() as session:
        await get_trip_or_404(session, trip_id)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in PHOTO_MEDIA_TYPES:
            ext = ".jpg"
        fname = f"t{trip_id}_{uuid.uuid4().hex[:10]}{ext}"
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        (PHOTO_DIR / fname).write_bytes(data)

        photo = Photo(
            trip_id=trip_id,
            filename=fname,
            lat=lat,
            lng=lng,
            timestamp=db.to_naive_utc(timestamp) if timestamp else db.utcnow(),
        )
        session.add(photo)
        await session.commit()
        print(f"[photos] trip {trip_id}: {fname} ({len(data)} bytes)")
        return {"photo_id": photo.id, "url": f"/photos/{photo.id}"}


@app.get("/photos/{photo_id}")
async def get_photo(photo_id: int):
    async with db.SessionLocal() as session:
        photo = (
            await session.execute(select(Photo).where(Photo.id == photo_id))
        ).scalar_one_or_none()
    if photo is None:
        raise HTTPException(status_code=404, detail=f"unknown photo_id {photo_id}")
    path = PHOTO_DIR / photo.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="photo file missing on disk")
    media = PHOTO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


@app.get("/photos/{photo_id}/board")
async def get_photo_board(photo_id: int, w: int = 480):
    """M55M1 用的縮圖：手機原圖幾 MB、可能是 progressive JPEG，
    tjpgd 解不動也塞不下——這裡縮到板子尺寸並強制 baseline JPEG。"""
    w = max(64, min(w, 1024))
    async with db.SessionLocal() as session:
        photo = (
            await session.execute(select(Photo).where(Photo.id == photo_id))
        ).scalar_one_or_none()
    if photo is None:
        raise HTTPException(status_code=404, detail=f"unknown photo_id {photo_id}")
    src = PHOTO_DIR / photo.filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="photo file missing on disk")

    cache = PHOTO_DIR / f"board_{photo_id}_{w}.jpg"
    if not cache.exists():
        def _resize():
            from PIL import Image

            img = Image.open(src).convert("RGB")
            img.thumbnail((w, w))
            img.save(cache, "JPEG", quality=85, progressive=False)

        try:
            await asyncio.to_thread(_resize)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"resize failed: {e}")
    return FileResponse(cache, media_type="image/jpeg")


# ================================================================ App: story
# 用旅程「真實資料」生成遊記（改造 Day 3 /generate：假資料 → 查 DB）。
# 素材有三：行程統計（日期/時長/距離）、實際經過的地點（重用景點候選池的
# 真實地名）、以及旅程照片（縮圖後走 vision 模型讓 LLM「看」照片內容）。
# 地點與照片都是 best-effort：查不到/vision 不支援就退回較簡單的版本，不擋生成。

WEEKDAY_ZH = "一二三四五六日"


def llm_json(prompt: str, image_uris: list[str] | None = None) -> dict:
    """One-shot LLM call returning parsed JSON. Same routing/env as call_llm.

    image_uris: data:image/... URIs → 走多模態（vision）路線，讓模型「看」照片。
    需要模型與 gateway 支援 vision（gemini-3-flash 支援）；不支援時 caller 應
    fallback 到純文字。"""
    key_vars = required_api_keys(LLM_MODEL)
    if not any(os.environ.get(k) for k in key_vars):
        raise HTTPException(
            status_code=500,
            detail=f"none of {key_vars} is set for model {LLM_MODEL} "
            "(or use LLM_MOCK=1 for pipeline testing)",
        )
    extra = {}
    if LLM_API_BASE:
        extra["api_base"] = LLM_API_BASE
    if os.environ.get("LLM_API_KEY"):
        extra["api_key"] = os.environ["LLM_API_KEY"]
    if LLM_REASONING_EFFORT:
        extra["reasoning_effort"] = LLM_REASONING_EFFORT
        extra["allowed_openai_params"] = ["reasoning_effort"]

    if image_uris:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": u}} for u in image_uris
        ]
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.8,
            timeout=45 if image_uris else 30,  # vision 較慢
            **extra,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


def build_trip_story_prompt(
    trip: Trip,
    visited: list[tuple[str, str]],
    passed: list[str],
    has_photos: bool,
) -> str:
    parts = [f"行程名稱:{trip.title or '未命名旅程'}"]
    if trip.start_time:
        local = trip.start_time.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        parts.append(
            f"日期:{local.year}年{local.month}月{local.day}日"
            f"(星期{WEEKDAY_ZH[local.weekday()]})"
        )
        parts.append(f"出發時間:{local.strftime('%H:%M')}")
        if trip.end_time:
            mins = max(1, int((trip.end_time - trip.start_time).total_seconds() // 60))
            parts.append(f"全程約:{mins} 分鐘")
    if trip.distance_m >= 1000:
        parts.append(f"走了約:{trip.distance_m / 1000:.1f} 公里")
    elif trip.distance_m > 0:
        parts.append(f"走了約:{trip.distance_m:.0f} 公尺")
    if visited:
        parts.append(
            "真的停留過的地方:"
            + "、".join(f"{name}（{note}）" for name, note in visited)
        )
    if passed:
        parts.append("沿路走過門口的:" + "、".join(passed))

    lines = [
        "你是家庭旅遊回憶 App「憶起」的遊記寫手。根據以下真實行程資料"
        + ("與隨附的旅程照片" if has_photos else "")
        + "，寫一段給家人回味的遊記。",
        "\n".join(parts),
        "\n要求：",
        "1. story：繁體中文 100 到 150 字，第一人稱「我們」，"
        "語氣溫暖親切，像寫在家庭相簿裡的回憶。",
    ]
    n = 2
    if visited:
        lines.append(
            f"{n}. 以「真的停留過的地方」為回憶主軸（停留時間、拍照都是真實線索）。"
        )
        n += 1
    if passed:
        lines.append(
            f"{n}. 「沿路走過門口的」只是路過，最多輕輕帶一兩個，"
            "不要寫成有進去消費或參觀。"
        )
        n += 1
    lines.append(
        f"{n}. 除了上面列出的地點，不要編造任何其他地名、店名或事件"
        + ("。" if (visited or passed) else
           "；沒有提供地名就不要提地名，改寫時間帶的氛圍、步行的心情與陪伴的感覺。")
    )
    n += 1
    if has_photos:
        lines.append(
            f"{n}. 參考照片裡真實看得到的內容（風景、食物、人物氛圍…）融入回憶，"
            "只描述照片中確實有的東西，不要無中生有。"
        )
        n += 1
    lines.append(f"{n}. 不要標題、不要引號，直接是遊記內容。")
    lines.append('只輸出 JSON：{"story": "..."}')
    return "\n".join(lines)


def photo_data_uri(path: Path, max_px: int = 768) -> str | None:
    """把照片檔縮圖成 <=max_px 的 JPEG，轉 data URI 供 vision 模型讀。
    讀不開（缺檔、HEIC 無解碼器…）回 None，該張略過。"""
    import base64
    import io

    try:
        from PIL import Image

        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # noqa: BLE001
        print(f"[story] photo encode failed {path.name}: {e}")
        return None


def select_photo_uris(filenames: list[str], k: int = 4) -> list[str]:
    """沿旅程時間軸均勻挑最多 k 張照片，編碼成 data URI。"""
    if not filenames:
        return []
    if len(filenames) <= k:
        chosen = filenames
    else:
        step = len(filenames) / k
        chosen = [filenames[int(i * step)] for i in range(k)]
    uris = [photo_data_uri(PHOTO_DIR / f) for f in chosen]
    return [u for u in uris if u]


def find_stays(
    track: list[tuple[float, float, datetime]],
    min_stay_s: float = 180,
    max_move_m: float = 80,
) -> list[tuple[float, float, float]]:
    """從軌跡找「停留點」：GPS 用 distanceFilter（10m）記錄，人停下來就不會
    產生新點——所以連續兩點時間差大、空間距離卻很近，就代表在那裡停留過。
    回 (lat, lng, 停留秒數)。"""
    stays = []
    for (lat1, lng1, t1), (lat2, lng2, t2) in zip(track, track[1:]):
        dt = (t2 - t1).total_seconds()
        if dt >= min_stay_s and db.haversine_m(lat1, lng1, lat2, lng2) <= max_move_m:
            stays.append(((lat1 + lat2) / 2, (lng1 + lng2) / 2, dt))
    return stays


def nearest_poi(pool: list[dict], lat: float, lng: float, max_m: float) -> dict | None:
    best, best_d = None, max_m
    for s in pool:
        d = db.haversine_m(lat, lng, s["lat"], s["lng"])
        if d <= best_d:
            best, best_d = s, d
    return best


def classify_visited(
    pool: list[dict],
    track: list[tuple[float, float, datetime]],
    photo_locs: list[tuple[float, float]],
    passed_max_m: float = 40,
    passed_cap: int = 6,
) -> tuple[list[tuple[str, str]], list[str]]:
    """把候選 POI 分成兩級證據：
      visited —— 停留點 100m 內（附「停留了約N分鐘」）或拍照位置 80m 內
                 （附「在這裡拍了照」）：幾乎確定真的有去。
      passed  —— 離軌跡 40m 內但沒停留：真的走過門口。
    500m 內其他的候選一概不給 LLM（那些只是「附近」，不是「去過」）。"""
    visited: dict[str, str] = {}
    for lat, lng, dwell in find_stays(track):
        poi = nearest_poi(pool, lat, lng, 100)
        if poi and poi["name"] not in visited:
            visited[poi["name"]] = f"停留了約 {max(1, round(dwell / 60))} 分鐘"
    for lat, lng in photo_locs:
        poi = nearest_poi(pool, lat, lng, 80)
        if poi:
            if poi["name"] in visited:
                if "拍了照" not in visited[poi["name"]]:
                    visited[poi["name"]] += "、還拍了照"
            else:
                visited[poi["name"]] = "在這裡拍了照"

    passed: list[tuple[float, str]] = []
    for s in pool:
        if s["name"] in visited:
            continue
        d = min(db.haversine_m(s["lat"], s["lng"], a, b) for a, b, _ in track)
        if d <= passed_max_m:
            passed.append((d, s["name"]))
    passed.sort()
    return list(visited.items()), [n for _, n in passed[:passed_cap]]


async def trip_visited_places(
    trip_id: int,
    track: list[tuple[float, float, datetime]],
    photo_locs: list[tuple[float, float]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """(visited, passed)；Overpass 失敗回空（best-effort，不擋遊記）。"""
    if not track:
        return [], []
    try:
        # 遊記的地點只是加分：短預算 best-effort，Overpass 慢/掛就跳過。
        pool = await build_spot_pool(
            trip_id, [(lat, lng) for lat, lng, _ in track], budget_s=15
        )
    except HTTPException as e:
        print(f"[story] spot lookup skipped: {e.detail}")
        return [], []
    return classify_visited(pool, track, photo_locs)


class StoryUpdate(BaseModel):
    story_text: str


@app.post("/trips/{trip_id}/story/generate")
async def generate_trip_story(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        # 趁 session 開著取出需要的資料
        track = [(p.lat, p.lng, p.timestamp) for p in trip.points]
        photo_files = [ph.filename for ph in trip.photos]
        photo_locs = [
            (ph.lat, ph.lng) for ph in trip.photos
            if ph.lat is not None and ph.lng is not None
        ]

        visited, passed = await trip_visited_places(trip_id, track, photo_locs)

        if os.environ.get("LLM_MOCK") == "1":
            if visited:
                where = f"在{visited[0][0]}{visited[0][1]}"
            elif passed:
                where = f"路過{passed[0]}"
            else:
                where = "在這附近散步"
            pic = "、拍了幾張照片" if photo_files else ""
            story = (
                f"這天我們一起出門走了走，{where}{pic}，"
                f"{trip.title or '這趟小旅程'}雖然不長，沿路的風景和彼此的笑聲，"
                "都讓平凡的一天變得暖暖的。回家的路上大家都說，下次還要再一起出來走走。"
            )
        else:
            t0 = time.time()
            image_uris = await asyncio.to_thread(select_photo_uris, photo_files)
            prompt = build_trip_story_prompt(trip, visited, passed, bool(image_uris))
            try:
                data = await asyncio.to_thread(llm_json, prompt, image_uris or None)
            except HTTPException:
                if not image_uris:
                    raise
                # vision 可能不被支援：退回純文字（仍帶真實地點）
                print("[story] vision failed, falling back to text-only")
                text_prompt = build_trip_story_prompt(trip, visited, passed, False)
                data = await asyncio.to_thread(llm_json, text_prompt)
            story = str(data.get("story", "")).strip()
            print(f"[story] trip {trip_id} llm={int((time.time() - t0) * 1000)}ms "
                  f"visited={len(visited)} passed={len(passed)} "
                  f"photos={len(image_uris)}")
            if not story:
                raise HTTPException(status_code=502, detail="LLM returned empty story")

        trip.story_text = story
        await session.commit()
        print(f"[story] trip {trip_id}: {story[:60]}...")
        return {"trip_id": trip_id, "story_text": story}


@app.put("/trips/{trip_id}/story")
async def update_trip_story(trip_id: int, req: StoryUpdate):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
        trip.story_text = req.story_text.strip()
        await session.commit()
        return {"trip_id": trip_id, "story_text": trip.story_text}


# --- 每旅程的板子用媒體檔（M55M1 SD 卡同步用） -----------------------------
# 以遊記內容 hash 當快取 key：遊記沒變就直接回檔案，變了自動重做。
# TTS 較慢（~5-10s）用鎖避免同檔並發重做；.tim 渲染快、同步做即可。

MEDIA_DIR = Path(__file__).parent / "trip_media"
MEDIA_DIR.mkdir(exist_ok=True)

_MEDIA_LOCK = asyncio.Lock()  # 一次一個 TTS，避免並發打 edge-tts


def _story_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


async def _load_trip_story(trip_id: int) -> tuple[Trip, str, str]:
    """回 (trip, title, subheader)；沒遊記丟 404。"""
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
    if not trip.story_text:
        raise HTTPException(status_code=404, detail="這趟旅程還沒有遊記")
    title = trip.title or "未命名旅程"
    subheader = ""
    if trip.start_time:
        local = trip.start_time.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        subheader = f"{local.year}年{local.month}月{local.day}日"
    return trip, title, subheader


@app.get("/trips/{trip_id}/story.txt")
async def trip_story_txt(trip_id: int):
    trip, _, _ = await _load_trip_story(trip_id)
    return PlainTextResponse(trip.story_text, media_type="text/plain; charset=utf-8")


@app.get("/trips/{trip_id}/story.tim")
async def trip_story_tim(trip_id: int):
    """LCD 文字圖（TIM4 4-bpp）——MCU 沒中文字型，顯示遊記靠這張。"""
    trip, title, subheader = await _load_trip_story(trip_id)
    f = MEDIA_DIR / f"t{trip_id}_{_story_hash(trip.story_text)}.tim"
    if not f.exists():
        tim4 = render_story_tim4(
            trip.story_text, header=title, subheader=subheader,
            preview_png=f.with_suffix(".png"),
        )
        f.write_bytes(tim4)
    return FileResponse(f, media_type="application/octet-stream")


@app.get("/trips/{trip_id}/story.wav")
async def trip_story_wav(trip_id: int):
    """遊記 TTS 語音（16kHz/16bit/mono WAV）。第一次要幾秒，之後走快取。"""
    trip, _, _ = await _load_trip_story(trip_id)
    f = MEDIA_DIR / f"t{trip_id}_{_story_hash(trip.story_text)}.wav"
    if not f.exists():
        async with _MEDIA_LOCK:
            if not f.exists():  # 等鎖期間可能已被別的請求做好
                try:
                    await tts_to_wav(trip.story_text, f)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
                print(f"[media] trip {trip_id} wav ready: {f.stat().st_size} bytes")
    return FileResponse(f, media_type="audio/wav", filename=f"t{trip_id}.wav")


# ================================================================ App: spots
# 沿途景點（App Stage 4）：沿路線取樣點 → Overpass API 查周邊「真實 POI」
# （餐廳/咖啡/景點/古蹟/公園…）→ 依離路線距離排序 → LLM 各寫一句推薦語。
#
# 為什麼用 Overpass 而非 Nominatim reverse：reverse 只回「座標點正上方是什麼」，
# 走路時座標落在馬路上，結果全是街道名。Overpass 能用 around: 半徑查附近有名字
# 的 POI，才問得出「附近有什麼好吃好玩的」。單一 union query 一次查完所有取樣點，
# 對 Overpass 公用實例友善（只發一個請求）。

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
OVERPASS_UA = "yiki-app/0.1 (family trip memory PoC; q56141036@gs.ncku.edu.tw)"
SPOT_RADIUS_M = 500   # 每個取樣點周邊搜尋半徑
SPOT_LIMIT = 12       # 最多回傳幾個景點

# OSM tag → 中文分類；沒對應的（純道路、住宅…）直接濾掉，這就是過濾「街道」的關鍵。
_AMENITY_KIND = {
    "restaurant": "餐廳", "food_court": "美食廣場", "cafe": "咖啡廳",
    "fast_food": "小吃", "bar": "酒吧", "pub": "居酒屋", "ice_cream": "冰品",
    "marketplace": "市場",
}
_TOURISM_KIND = {
    "attraction": "景點", "museum": "博物館", "viewpoint": "觀景點",
    "artwork": "公共藝術", "gallery": "藝廊", "theme_park": "樂園",
    "zoo": "動物園", "aquarium": "水族館", "picnic_site": "野餐地",
}
_LEISURE_KIND = {"park": "公園", "garden": "庭園"}

# 用來把景點和美食平衡穿插，避免觀光區餐廳把古蹟/景點洗版。
FOOD_KINDS = set(_AMENITY_KIND.values()) | {"烘焙坊"}


def poi_kind(tags: dict) -> str | None:
    """OSM tags → 中文分類標籤；不是我們想推薦的類型回 None（濾掉）。"""
    if tags.get("amenity") in _AMENITY_KIND:
        return _AMENITY_KIND[tags["amenity"]]
    if tags.get("tourism") in _TOURISM_KIND:
        return _TOURISM_KIND[tags["tourism"]]
    if tags.get("historic"):
        return "古蹟"
    if tags.get("leisure") in _LEISURE_KIND:
        return _LEISURE_KIND[tags["leisure"]]
    if tags.get("shop") == "bakery":
        return "烘焙坊"
    return None


# 兩層快取，都以 point_count 當版本（點數變了就失效）：
#   _SPOTS_POOL   trip_id -> (pc, 全部候選 POI)。重用可免重打 Overpass，
#                 「重新推薦」也吃這層，只換選出來的子集，不再打一次 Overpass。
#   _SPOTS_RESULT trip_id -> (pc, 上次回傳的清單)。非 refresh 時直接回，秒開。
_SPOTS_POOL: dict[int, tuple[int, list[dict]]] = {}
_SPOTS_RESULT: dict[int, tuple[int, list[dict]]] = {}


def sample_route_points(
    pts: list[tuple[float, float]], max_k: int = 5, min_gap_m: float = 250
) -> list[tuple[float, float]]:
    """沿路線均勻取最多 max_k 個點，彼此至少相距 min_gap_m。"""
    if len(pts) <= max_k:
        cand = pts
    else:
        step = (len(pts) - 1) / (max_k - 1)
        cand = [pts[round(i * step)] for i in range(max_k)]
    reps: list[tuple[float, float]] = []
    for c in cand:
        if all(db.haversine_m(c[0], c[1], r[0], r[1]) > min_gap_m for r in reps):
            reps.append(c)
    return reps or [pts[0]]


def build_overpass_query(
    points: list[tuple[float, float]], radius_m: int = SPOT_RADIUS_M
) -> str:
    filters = [
        '["amenity"~"^(restaurant|cafe|fast_food|bar|pub|ice_cream|food_court|marketplace)$"]',
        '["tourism"~"^(attraction|museum|viewpoint|artwork|gallery|theme_park|zoo|aquarium|picnic_site)$"]',
        '["historic"]["name"]',
        '["leisure"~"^(park|garden)$"]["name"]',
        '["shop"="bakery"]',
    ]
    clauses = "".join(
        f"nwr{f}(around:{radius_m},{lat},{lng});"
        for lat, lng in points
        for f in filters
    )
    return f"[out:json][timeout:25];({clauses});out center tags 120;"


def parse_overpass_elements(
    elements: list[dict], ref_pts: list[tuple[float, float]]
) -> list[dict]:
    """Overpass elements → 候選 POI（分類、離 ref_pts 最近距離、名稱去重）。"""
    candidates: list[dict] = []
    seen: set[str] = set()
    for el in elements:
        tags = el.get("tags") or {}
        kind = poi_kind(tags)
        if kind is None:
            continue
        name = (
            tags.get("name:zh-TW") or tags.get("name:zh")
            or tags.get("name") or tags.get("int_name")
        )
        if not name or name in seen:
            continue
        if el.get("type") == "node":
            slat, slng = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            slat, slng = center.get("lat"), center.get("lon")
        if slat is None or slng is None:
            continue
        seen.add(name)
        candidates.append({
            "name": name,
            "lat": slat,
            "lng": slng,
            "distance_m": round(
                min(db.haversine_m(slat, slng, a, b) for a, b in ref_pts), 1
            ),
            "category": kind,
            "description": "",
        })
    return candidates


async def overpass_pois(
    client: httpx.AsyncClient, points: list[tuple[float, float]],
    total_budget_s: float = 40, radius_m: int = SPOT_RADIUS_M,
) -> list[dict]:
    """查沿線周邊 POI；回原始 element 清單。任一鏡像成功即回。
    total_budget_s：所有鏡像加起來的時間上限，避免整批過載時無止境地等
    （遊記那邊給較短的預算，查不到就跳過，不拖住生成）。"""
    query = build_overpass_query(points, radius_m)
    start = time.monotonic()
    last_err: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        remaining = total_budget_s - (time.monotonic() - start)
        if remaining < 3:  # 剩沒幾秒就別再試下一個鏡像
            break
        try:
            r = await client.post(
                url, data={"data": query},
                headers={"User-Agent": OVERPASS_UA},
                timeout=min(20, remaining),
            )
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as e:  # noqa: BLE001 — 換下一個鏡像
            last_err = e
            print(f"[spots] overpass {url} failed: {e}")
    raise HTTPException(status_code=502, detail=f"景點服務暫時無法使用：{last_err}")


def balanced_pick(spots: list[dict], limit: int, shuffle: bool = False) -> list[dict]:
    """景點與美食各按距離排序後交錯取 limit 個，避免餐廳把景點洗版；
    某一類不夠就用另一類補滿。最後再依距離排序供顯示。

    shuffle=True（重新推薦）：各類先取最近的一批候選再打亂，換一組「附近但
    不同」的地點，而不是每次都回一模一樣的最近幾個。"""
    food = sorted((s for s in spots if s["category"] in FOOD_KINDS),
                  key=lambda s: s["distance_m"])
    sight = sorted((s for s in spots if s["category"] not in FOOD_KINDS),
                   key=lambda s: s["distance_m"])
    if shuffle:
        food = food[: limit * 2]
        sight = sight[: limit * 2]
        random.shuffle(food)
        random.shuffle(sight)
    out: list[dict] = []
    fi = si = 0
    # 從比較近的那一類先開始，之後每次換邊
    take_food = bool(food) and (not sight or food[0]["distance_m"] <= sight[0]["distance_m"])
    while len(out) < limit and (fi < len(food) or si < len(sight)):
        if take_food and fi < len(food):
            out.append(food[fi]); fi += 1
        elif not take_food and si < len(sight):
            out.append(sight[si]); si += 1
        elif fi < len(food):
            out.append(food[fi]); fi += 1
        elif si < len(sight):
            out.append(sight[si]); si += 1
        take_food = not take_food
    return sorted(out, key=lambda s: s["distance_m"])


def build_spot_lines_prompt(spots: list[dict]) -> str:
    listing = "\n".join(
        f"{i + 1}. {s['name']}（{s['category']}）" for i, s in enumerate(spots)
    )
    return (
        "你是旅遊 App「憶起」的在地嚮導。使用者這趟走路經過的周邊有以下地點，"
        "為每個地點各寫一句 15 到 25 字的繁體中文推薦語。\n"
        f"{listing}\n\n"
        "要求：語氣溫暖口語，像在地朋友的順路推薦；美食就點出想嚐的感覺、"
        "景點就講值得停留的理由。只依名稱與分類合理描述，不確定的細節不要編造。\n"
        '只輸出 JSON：{"spots": [{"name": "地點名稱", "line": "推薦語"}]}'
    )


async def build_spot_pool(trip_id: int, pts: list[tuple[float, float]],
                          budget_s: float = 40) -> list[dict]:
    """查 Overpass 並解析成候選 POI 池（含分類、離路線距離），依名稱去重。
    結果快取起來，重新推薦時重用、不再打 Overpass。
    budget_s：Overpass 總時間上限（遊記那邊給短預算，best-effort）。"""
    pool = _SPOTS_POOL.get(trip_id)
    if pool and pool[0] == len(pts):
        return pool[1]

    samples = sample_route_points(pts)
    async with httpx.AsyncClient() as client:
        elements = await overpass_pois(client, samples, total_budget_s=budget_s)

    candidates = parse_overpass_elements(elements, pts)
    _SPOTS_POOL[trip_id] = (len(pts), candidates)
    print(f"[spots] trip {trip_id}: pool of {len(candidates)} POIs "
          f"from {len(samples)} sample points")
    return candidates


# --- 記錄中的「附近有什麼」（以目前位置為中心，走路時用） -------------------
# 兩段式：預設不跑 LLM 先快回清單；App 再帶 describe=true 補 AI 介紹。
# 以 ~100m 網格 + TTL 快取（含「已寫過介紹」旗標），連按不重打 Overpass/LLM。

NEARBY_RADIUS_M = 400
NEARBY_TTL_S = 600
# key -> (monotonic, spots, described)
_NEARBY_CACHE: dict[tuple[float, float], tuple[float, list[dict], bool]] = {}


async def _describe_spots(spots: list[dict]) -> None:
    """就地補上 LLM 推薦語；失敗只印 log、介紹留空（best-effort）。"""
    if not spots:
        return
    if os.environ.get("LLM_MOCK") == "1":
        for s in spots:
            s["description"] = f"順路的{s['category']}，經過時不妨停下來看看。"
        return
    try:
        data = await asyncio.to_thread(llm_json, build_spot_lines_prompt(spots))
        lines = {
            d.get("name"): str(d.get("line", "")).strip()
            for d in data.get("spots", [])
            if isinstance(d, dict)
        }
        for s in spots:
            s["description"] = lines.get(s["name"], "")
    except HTTPException as e:
        print(f"[spots] nearby LLM lines skipped: {e.detail}")


@app.get("/spots/nearby")
async def nearby_spots(lat: float, lng: float, describe: bool = False):
    key = (round(lat, 3), round(lng, 3))
    hit = _NEARBY_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < NEARBY_TTL_S:
        ts, spots, described = hit
        if describe and not described:
            await _describe_spots(spots)
            _NEARBY_CACHE[key] = (ts, spots, True)
        return {"spots": spots, "cached": True}

    async with httpx.AsyncClient() as client:
        elements = await overpass_pois(
            client, [(lat, lng)], total_budget_s=25, radius_m=NEARBY_RADIUS_M
        )
    spots = balanced_pick(parse_overpass_elements(elements, [(lat, lng)]), 10)
    if describe:
        await _describe_spots(spots)
    _NEARBY_CACHE[key] = (time.monotonic(), spots, describe)
    print(f"[spots] nearby ({lat:.4f},{lng:.4f}): {len(spots)} POIs "
          f"(describe={describe})")
    return {"spots": spots, "cached": False}


@app.get("/trips/{trip_id}/spots")
async def trip_spots(trip_id: int, refresh: bool = False):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        pts = [(p.lat, p.lng) for p in trip.points]
    if not pts:
        raise HTTPException(status_code=400, detail="這趟旅程沒有軌跡點，無法推薦景點")

    # 非 refresh：有上次結果就直接回（再次進頁面秒開）。refresh 一律重選。
    if not refresh:
        done = _SPOTS_RESULT.get(trip_id)
        if done and done[0] == len(pts):
            return {"spots": done[1], "cached": True}

    candidates = await build_spot_pool(trip_id, pts)
    if not candidates:
        # Overpass 有回應但附近沒有 POI（偏遠路段）：回空清單，App 顯示友善訊息。
        _SPOTS_RESULT[trip_id] = (len(pts), [])
        return {"spots": [], "cached": False}

    spots = balanced_pick(candidates, SPOT_LIMIT, shuffle=refresh)

    # LLM 推薦語是加分項：失敗就留空，不讓整個請求掛掉。
    # 每次都重跑（temperature 0.8），重新推薦時連文案也會換句話講。
    if os.environ.get("LLM_MOCK") == "1":
        for s in spots:
            s["description"] = f"順路的{s['category']}，經過時不妨停下來看看。"
    else:
        try:
            data = await asyncio.to_thread(llm_json, build_spot_lines_prompt(spots))
            lines = {
                d.get("name"): str(d.get("line", "")).strip()
                for d in data.get("spots", [])
                if isinstance(d, dict)
            }
            for s in spots:
                s["description"] = lines.get(s["name"], "")
        except HTTPException as e:
            print(f"[spots] LLM lines skipped: {e.detail}")

    _SPOTS_RESULT[trip_id] = (len(pts), spots)
    print(f"[spots] trip {trip_id}: returned {len(spots)} POIs (refresh={refresh})")
    return {"spots": spots, "cached": False}


# --- 收藏進旅程的景點 -------------------------------------------------------
# 使用者記錄中在「附近景點」按 ➕ 收藏 → 存 trip_spots 表；
# 旅程詳情顯示的就是這份親手選過的清單（不再自動推薦）。


class SpotSaveRequest(BaseModel):
    name: str
    lat: float
    lng: float
    category: str = ""
    description: str = ""


def saved_spot_json(s: TripSpot, track: list[tuple[float, float]]) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "lat": s.lat,
        "lng": s.lng,
        "distance_m": round(
            min(db.haversine_m(s.lat, s.lng, a, b) for a, b in track), 1
        ) if track else 0,
        "category": s.category,
        "description": s.description,
    }


@app.post("/trips/{trip_id}/spots/save")
async def save_trip_spot(trip_id: int, req: SpotSaveRequest):
    async with db.SessionLocal() as session:
        await get_trip_or_404(session, trip_id)
        existing = (
            await session.execute(
                select(TripSpot).where(
                    TripSpot.trip_id == trip_id, TripSpot.name == req.name
                )
            )
        ).scalar_one_or_none()
        if existing:
            return {"spot_id": existing.id, "duplicate": True}
        spot = TripSpot(
            trip_id=trip_id,
            name=req.name.strip(),
            lat=req.lat,
            lng=req.lng,
            category=req.category.strip(),
            description=req.description.strip(),
        )
        session.add(spot)
        await session.commit()
        print(f"[spots] trip {trip_id} saved: {spot.name}")
        return {"spot_id": spot.id, "duplicate": False}


@app.get("/trips/{trip_id}/spots/saved")
async def list_saved_spots(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)
        track = [(p.lat, p.lng) for p in trip.points]
        spots = (
            await session.execute(
                select(TripSpot)
                .where(TripSpot.trip_id == trip_id)
                .order_by(TripSpot.created_at)
            )
        ).scalars().all()
        return {"spots": [saved_spot_json(s, track) for s in spots]}


@app.delete("/trips/{trip_id}/spots/saved/{spot_id}")
async def delete_saved_spot(trip_id: int, spot_id: int):
    async with db.SessionLocal() as session:
        spot = (
            await session.execute(
                select(TripSpot).where(
                    TripSpot.id == spot_id, TripSpot.trip_id == trip_id
                )
            )
        ).scalar_one_or_none()
        if spot is None:
            raise HTTPException(status_code=404, detail=f"unknown spot_id {spot_id}")
        await session.delete(spot)
        await session.commit()
        return {"deleted": spot_id}


# ================================================================ App: frames
# 相框同步（App Stage 5，整批模式）：App 用配對碼配對；M55M1 輪詢
# GET /frames/{id}/sync 拿「全部旅程」的檔案清單，逐檔下載到 SD 卡、
# 以資料夾區分（TRIPS/T0001/STORY.TXT+STORY.TIM+STORY.WAV+Pxxxx.JPG）。
# 每趟旅程有 version（遊記+照片 hash），韌體存進 VERSION.TXT，
# 下次同步比對相同就整趟跳過——只下載有變動的旅程。
# 檔名全走 8.3 格式（FatFS 安全）。


class PairRequest(BaseModel):
    pair_code: str


def frame_json(frame: Frame, trip_count: int) -> dict:
    return {
        "frame_id": frame.id,
        "name": frame.name,
        "last_sync": db.iso_z(frame.last_sync),
        "trip_count": trip_count,
    }


async def get_frame_or_404(session, frame_id: int) -> Frame:
    frame = (
        await session.execute(select(Frame).where(Frame.id == frame_id))
    ).scalar_one_or_none()
    if frame is None:
        raise HTTPException(status_code=404, detail=f"unknown frame_id {frame_id}")
    return frame


async def _trip_count(session) -> int:
    return len((await session.execute(select(Trip.id))).scalars().all())


@app.post("/frames/pair")
async def pair_frame(req: PairRequest):
    code = req.pair_code.strip()
    async with db.SessionLocal() as session:
        frame = (
            await session.execute(select(Frame).where(Frame.pair_code == code))
        ).scalar_one_or_none()
        if frame is None:
            raise HTTPException(status_code=404, detail="配對碼錯誤，請確認相框螢幕上的 6 位數字")
        print(f"[frames] paired frame {frame.id} ({frame.name})")
        return frame_json(frame, await _trip_count(session))


@app.get("/frames/{frame_id}")
async def frame_status(frame_id: int):
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        return frame_json(frame, await _trip_count(session))


@app.get("/frames/{frame_id}/sync")
async def frame_sync(frame_id: int):
    """M55M1 同步清單：所有「有內容」（遊記或照片）的旅程與其檔案 URL。
    範例韌體流程：
      for trip in items:
        if SD:/TRIPS/{folder}/VERSION.TXT == trip.version: continue  # 沒變
        下載 story_txt/story_tim/story_wav 與 photos[].url 進該資料夾
        寫 VERSION.TXT
    """
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        trips = (
            (await session.execute(
                select(Trip)
                .options(selectinload(Trip.photos))
                .order_by(Trip.created_at)
            )).scalars().all()
        )

        items = []
        for trip in trips:
            if not trip.story_text and not trip.photos:
                continue  # 空旅程不佔相框空間
            date_str = ""
            if trip.start_time:
                local = trip.start_time.replace(
                    tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
                date_str = f"{local.year}-{local.month:02d}-{local.day:02d}"
            version = hashlib.md5(
                (trip.story_text + "|" + trip.title + "|"
                 + ",".join(str(ph.id) for ph in trip.photos)).encode("utf-8")
            ).hexdigest()[:8]
            has_story = bool(trip.story_text)
            items.append({
                "trip_id": trip.id,
                "folder": f"T{trip.id:04d}",
                "title": trip.title or "未命名旅程",
                "date": date_str,
                "version": version,
                "has_story": has_story,
                "story_txt": f"/trips/{trip.id}/story.txt" if has_story else None,
                "story_tim": f"/trips/{trip.id}/story.tim" if has_story else None,
                "story_wav": f"/trips/{trip.id}/story.wav" if has_story else None,
                "photos": [
                    {"name": f"P{ph.id:04d}.JPG", "url": f"/photos/{ph.id}/board"}
                    for ph in trip.photos
                ],
            })

        frame.last_sync = db.utcnow()
        await session.commit()
        print(f"[frames] frame {frame_id} sync manifest: {len(items)} trip(s)")
        return {"trips": items, "count": len(items)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
