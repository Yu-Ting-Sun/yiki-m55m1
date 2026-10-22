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
#   POST /frames/{id}/voice         -> push-to-talk with 小憶: 8 kHz mu-law in,
#                                      reply as a story-rail TIM4 out
#   GET  /frames/{id}/messages      -> messages the elder left through 小憶
#                                      (App polls; .../{mid}/audio = the voice)
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
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import litellm  # heavy import (~10 s); pay it at startup, not on first /generate
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

import db
from db import Frame, FrameMessage, GpsPoint, LikeEvent, Photo, Trip, TripSpot
from textimg import render_reply_tim4, render_story_tim4

# 從 backend/.env 載入環境變數（若存在）——必須在下面所有 os.environ.get(...)
# 之前跑。shell 已設的變數優先（override=False），不會被 .env 蓋掉。
load_dotenv(Path(__file__).with_name(".env"))


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
# 小憶的聲線微調(使用者選定):音高 +20Hz 更年輕、語速 +10% 更有精神。
# 影片旁白、App 語音、遊記朗讀三處共用,保持角色一致。
TTS_RATE = "+10%"
TTS_PITCH = "+20Hz"

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
        communicate = edge_tts.Communicate(
            text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
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
    # App 當時配對的相框；同步時只有這台（或未指定的舊旅程）會拿到這趟。
    frame_id: int | None = None


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
        trip = Trip(title=req.title.strip(), start_time=db.utcnow(),
                    frame_id=req.frame_id)
        session.add(trip)
        await session.commit()
        print(f"[trips] started trip {trip.id} (frame={req.frame_id})")
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
        detail["members"] = trip_members(trip)
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


@app.delete("/photos/{photo_id}")
async def delete_photo(photo_id: int):
    """刪掉一張旅程照片：DB 紀錄 + 磁碟原檔 + 板子縮圖快取。
    旅程 version 含照片 id 清單，刪除後版本改變 → 相框下次同步自動更新。"""
    async with db.SessionLocal() as session:
        photo = (
            await session.execute(select(Photo).where(Photo.id == photo_id))
        ).scalar_one_or_none()
        if photo is None:
            raise HTTPException(status_code=404, detail=f"unknown photo_id {photo_id}")
        fname = photo.filename
        await session.delete(photo)
        await session.commit()
    (PHOTO_DIR / fname).unlink(missing_ok=True)
    for cache in PHOTO_DIR.glob(f"board_{photo_id}_*.jpg"):
        cache.unlink(missing_ok=True)
    print(f"[photos] deleted photo {photo_id} ({fname})")
    return {"deleted": True}


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


class TitleUpdate(BaseModel):
    title: str


@app.put("/trips/{trip_id}/title")
async def update_trip_title(trip_id: int, req: TitleUpdate):
    title = req.title.strip()[:100]
    if not title:
        raise HTTPException(status_code=400, detail="標題不能是空的")
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
        trip.title = title
        await session.commit()
        return {"trip_id": trip_id, "title": title}


# --- 旅程參加者（相框 label.json 的資料來源） -------------------------------
# 板端限制（Slideshow.c）：每相簿最多 8 人、每個 label 最多 23 bytes、
# label.json 只解析前 512 bytes。label 需與人臉註冊（enroll_<label>.raw）一致。

FRAME_MAX_USERS = 8
FRAME_USER_MAX_BYTES = 23


def trip_members(trip: Trip) -> list[str]:
    try:
        v = json.loads(trip.members or "[]")
        return [str(m) for m in v] if isinstance(v, list) else []
    except ValueError:
        return []


class MembersUpdate(BaseModel):
    members: list[str]


@app.put("/trips/{trip_id}/members")
async def update_trip_members(trip_id: int, req: MembersUpdate):
    cleaned = []
    for m in req.members:
        m = m.strip()
        if not m:
            continue
        if len(m.encode("utf-8")) > FRAME_USER_MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"名字「{m}」太長（相框限制 {FRAME_USER_MAX_BYTES} bytes）",
            )
        if m not in cleaned:
            cleaned.append(m)
    if len(cleaned) > FRAME_MAX_USERS:
        raise HTTPException(
            status_code=400, detail=f"最多 {FRAME_MAX_USERS} 位參加者（相框限制）"
        )
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
        trip.members = json.dumps(cleaned, ensure_ascii=False)
        await session.commit()
        return {"trip_id": trip_id, "members": cleaned}


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
    # 快取 key 必須涵蓋「圖上渲染的全部內容」——標題/日期也在圖裡，
    # 只 hash 內文會讓改標題後繼續吐舊圖。
    f = MEDIA_DIR / (
        f"t{trip_id}_{_story_hash(trip.story_text + '|' + title + '|' + subheader)}.tim"
    )
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


@app.get("/trips/{trip_id}/story.ulaw")
async def trip_story_ulaw(trip_id: int):
    """遊記語音給相框用的格式：8 kHz G.711 mu-law，沒有檔頭。

    板子的 Wi-Fi 模組只有約 11.5 KB/s，16-bit WAV 一趟遊記要傳快一分鐘；
    mu-law 是四分之一大小（8 KB/s），板端解碼只是查表（Audio.c），音質
    對語音夠用。內容同 story.wav，所以沿用它的 TTS 快取。"""
    trip, _, _ = await _load_trip_story(trip_id)
    wav = MEDIA_DIR / f"t{trip_id}_{_story_hash(trip.story_text)}.wav"
    ulaw = wav.with_suffix(".ulaw")

    if not ulaw.exists():
        async with _MEDIA_LOCK:
            if not wav.exists():
                try:
                    await tts_to_wav(trip.story_text, wav)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
            if not ulaw.exists():
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", str(wav),
                    "-ar", "8000", "-ac", "1", "-f", "mulaw", str(ulaw),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    ulaw.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=500,
                        detail="mu-law encode failed: "
                               f"{stderr.decode(errors='replace')[-300:]}")
                print(f"[media] trip {trip_id} ulaw ready: "
                      f"{ulaw.stat().st_size} bytes")
    return FileResponse(ulaw, media_type="application/octet-stream",
                        filename=f"t{trip_id}.ulaw")


@app.get("/trips/{trip_id}/label.json")
async def trip_label_json(trip_id: int):
    """相簿參加者標籤——板端 Slideshow 的 naive parser 讀這個檔決定
    人臉辨識後要不要播這本相簿。格式固定 {"users": [...]}，遠小於
    板端 512-byte 解析上限。"""
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id)
    return Response(
        content=json.dumps({"users": trip_members(trip)}, ensure_ascii=False),
        media_type="application/json",
    )


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


# --- POI 供應層：Foursquare（設了 key 就用）或 Overpass（免費、預設） ---------
# 設 FOURSQUARE_API_KEY 就走 Foursquare Places（穩定、venue 資料好）；沒設或
# 失敗就退回 Overpass。兩者都回同一種 spot dict，分類標籤都對到 FOOD_KINDS
# / 非 FOOD_KINDS，balanced_pick 與 App 的 isFood 判斷才一致。

FOURSQUARE_API_KEY = os.environ.get("FOURSQUARE_API_KEY", "").strip()
FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"
FSQ_API_VERSION = "2025-06-17"

# 啟動時印出用哪個 POI 供應商，一眼可查（logs 開頭找 [poi]）。
print(f"[poi] provider: "
      f"{'Foursquare (key set)' if FOURSQUARE_API_KEY else 'Overpass (no key)'}")


def fsq_category(cat_name: str) -> tuple[bool, str] | None:
    """Foursquare 英文分類名 → (是否美食, 中文標籤)；認不得的回 None（濾掉，
    如停車場/銀行/辦公室）。美食標籤限 FOOD_KINDS 內、景點標籤在其外。"""
    n = cat_name.lower()
    # 美食
    if any(k in n for k in ("ice cream", "gelato", "frozen yogurt", "shaved ice")):
        return True, "冰品"
    if any(k in n for k in ("coffee", "café", "cafe", "tea", "bubble", "boba")):
        return True, "咖啡廳"
    if any(k in n for k in ("bakery", "pastry", "bagel", "donut", "bread")):
        return True, "烘焙坊"
    if any(k in n for k in ("bar", "pub", "brewery", "beer", "cocktail", "wine", "izakaya")):
        return True, "酒吧"
    if "market" in n and "supermarket" not in n:
        return True, "市場"
    if any(k in n for k in (
            "restaurant", "food", "diner", "dining", "noodle", "bbq", "grill",
            "steak", "pizza", "burger", "sushi", "ramen", "dumpling", "hotpot",
            "hot pot", "buffet", "snack", "breakfast", "brunch", "seafood",
            "dessert", "eatery", "bistro", "deli", "canteen", "joint", "curry")):
        return True, "餐廳"
    # 景點/地標
    if "museum" in n:
        return False, "博物館"
    if any(k in n for k in ("historic", "monument", "heritage", "castle",
                            "fort", "ruins", "memorial")):
        return False, "古蹟"
    if any(k in n for k in ("temple", "shrine", "church", "mosque")):
        return False, "廟宇"
    if any(k in n for k in ("park", "garden", "plaza", "square")):
        return False, "公園"
    if any(k in n for k in ("scenic", "lookout", "overlook", "viewpoint", "harbor", "pier", "beach")):
        return False, "觀景點"
    if any(k in n for k in ("art gallery", "gallery", "art museum", "theater", "cultural")):
        return False, "藝文"
    if any(k in n for k in ("landmark", "tourist", "attraction", "zoo", "aquarium")):
        return False, "景點"
    return None


async def foursquare_search(lat: float, lng: float, radius_m: int,
                            limit: int = 50,
                            query: str | None = None) -> list[dict]:
    """Foursquare Place Search → spot dict 清單（已濾掉非美食/景點）。
    query：文字搜尋（例「牛肉湯」），Foursquare 會比對店名/類別/tips。"""
    params = {
        "ll": f"{lat},{lng}",
        "radius": str(max(50, min(int(radius_m), 100000))),
        "limit": str(min(limit, 50)),
        "sort": "DISTANCE",
    }
    if query:
        params["query"] = query
    headers = {
        "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
        "X-Places-Api-Version": FSQ_API_VERSION,
        "accept": "application/json",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(FSQ_SEARCH_URL, params=params,
                             headers=headers, timeout=15)
    r.raise_for_status()
    out: list[dict] = []
    seen: set[str] = set()
    for p in r.json().get("results", []):
        plat, plng = p.get("latitude"), p.get("longitude")
        name = (p.get("name") or "").strip()
        if plat is None or plng is None or not name or name in seen:
            continue
        kind = None
        for c in p.get("categories", []):
            kind = fsq_category(c.get("name", ""))
            if kind:
                break
        if kind is None:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "lat": plat,
            "lng": plng,
            "distance_m": round(float(p.get("distance", 0)), 1),
            "category": kind[1],
            "description": "",
        })
    return out


POI_LAST_PROVIDER = {"name": "none"}  # 最近一次實際用到的供應商（觀測用）


async def find_pois(lat: float, lng: float, radius_m: int,
                    budget_s: float = 20,
                    query: str | None = None) -> list[dict]:
    """單點周邊 POI；有 Foursquare key 就用它，失敗或沒 key 退回 Overpass。
    query：具體品項的文字搜尋（Overpass 退路只能用名稱包含比對，best-effort）。"""
    if FOURSQUARE_API_KEY:
        try:
            spots = await foursquare_search(lat, lng, radius_m, query=query)
            POI_LAST_PROVIDER["name"] = "foursquare"
            return spots
        except Exception as e:  # noqa: BLE001
            print(f"[poi] foursquare failed ({e}); falling back to overpass")
    async with httpx.AsyncClient() as client:
        elements = await overpass_pois(
            client, [(lat, lng)], total_budget_s=budget_s, radius_m=int(radius_m))
    POI_LAST_PROVIDER["name"] = "overpass"
    spots = parse_overpass_elements(elements, [(lat, lng)])
    if query:
        spots = [s for s in spots if query.lower() in s["name"].lower()]
    return spots


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
    """沿路線取樣點查周邊 POI（Foursquare 或 Overpass）→ 候選池，依名稱去重、
    距離改成到最近軌跡點。結果快取，重新推薦時重用、不再重查。"""
    pool = _SPOTS_POOL.get(trip_id)
    if pool and pool[0] == len(pts):
        return pool[1]

    samples = sample_route_points(pts)
    merged: dict[str, dict] = {}
    for slat, slng in samples:
        try:
            for s in await find_pois(slat, slng, SPOT_RADIUS_M, budget_s=budget_s):
                merged.setdefault(s["name"], s)
        except Exception as e:  # noqa: BLE001 — 單點失敗略過，其他點照查
            print(f"[spots] sample ({slat:.4f},{slng:.4f}) failed: {e}")

    candidates = list(merged.values())
    for s in candidates:  # 距離統一成到最近軌跡點
        s["distance_m"] = round(
            min(db.haversine_m(s["lat"], s["lng"], a, b) for a, b in pts), 1)
    _SPOTS_POOL[trip_id] = (len(pts), candidates)
    print(f"[spots] trip {trip_id}: pool of {len(candidates)} POIs "
          f"from {len(samples)} sample points")
    return candidates


# --- 記錄中的「附近有什麼」（以目前位置為中心，走路時用） -------------------
# 兩段式：預設不跑 LLM 先快回清單；App 再帶 describe=true 補 AI 介紹。
# 以 ~100m 網格 + TTL 快取（含「已寫過介紹」旗標），連按不重打 Overpass/LLM。

NEARBY_RADIUS_M = 400
NEARBY_TTL_S = 600
# 兩層快取（同 grid key）：
#   _NEARBY_POOL   完整候選池——「重新推薦」換組時重用，不重打 Overpass
#   _NEARBY_RESULT 上次回傳的清單（含已寫介紹旗標），非 refresh 時秒回
_NEARBY_POOL: dict[tuple[float, float], tuple[float, list[dict]]] = {}
_NEARBY_RESULT: dict[tuple[float, float], tuple[float, list[dict], bool]] = {}


async def _nearby_pool(
    key: tuple[float, float], lat: float, lng: float
) -> list[dict]:
    hit = _NEARBY_POOL.get(key)
    if hit and time.monotonic() - hit[0] < NEARBY_TTL_S:
        return hit[1]
    candidates = await find_pois(lat, lng, NEARBY_RADIUS_M, budget_s=25)
    _NEARBY_POOL[key] = (time.monotonic(), candidates)
    return candidates


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
async def nearby_spots(
    lat: float, lng: float, describe: bool = False, refresh: bool = False
):
    key = (round(lat, 3), round(lng, 3))
    if not refresh:
        hit = _NEARBY_RESULT.get(key)
        if hit and time.monotonic() - hit[0] < NEARBY_TTL_S:
            ts, spots, described = hit
            if describe and not described:
                await _describe_spots(spots)
                _NEARBY_RESULT[key] = (ts, spots, True)
            return {"spots": spots, "cached": True}

    candidates = await _nearby_pool(key, lat, lng)
    # refresh：從最近的候選裡隨機換一組（美食/景點仍平衡），不重查
    spots = balanced_pick(candidates, 10, shuffle=refresh)
    if describe:
        await _describe_spots(spots)
    _NEARBY_RESULT[key] = (time.monotonic(), spots, describe)
    print(f"[spots] nearby ({lat:.4f},{lng:.4f}): {len(spots)} POIs "
          f"(refresh={refresh}, describe={describe}, "
          f"provider={POI_LAST_PROVIDER['name']})")
    return {"spots": spots, "cached": False,
            "provider": POI_LAST_PROVIDER["name"]}


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


# ================================================================ App: guide
# 導遊精靈「小憶」——LLM agent：判斷意圖 → 呼叫工具（附近 POI / 地點介紹）
# → 回溫暖口語，並附一個 action 標籤讓 App 的角色做動作（揮手/指路/說話）。
# 用「路由 + 組答」兩段 llm_json（不依賴 function-calling API，proxy 一定吃）。

GUIDE_WALK_M_PER_MIN = 80          # 走路速度估計（≈5 km/h）
GUIDE_BIKE_M_PER_MIN = 250         # 騎車估計（≈15 km/h）
GUIDE_DETOUR = 1.3                 # 直線 → 實際路程的粗略放大係數
GUIDE_MAX_RADIUS_M = 1500

# 地名 → 座標（forward geocoding）。Nominatim 使用政策：1 req/s + 自訂 UA。
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEO_UA = "yiki-app/0.1 (family trip memory PoC; q56141036@gs.ncku.edu.tw)"
_geo_last_call = 0.0


async def geocode_place(query: str, near: tuple[float, float] | None = None
                        ) -> tuple[float, float, str] | None:
    """地名查座標；near 提供時偏好附近的結果。查不到回 None。"""
    global _geo_last_call
    wait = 1.1 - (time.monotonic() - _geo_last_call)
    if wait > 0:
        await asyncio.sleep(wait)
    _geo_last_call = time.monotonic()

    params = {"q": query, "format": "jsonv2", "limit": "1",
              "accept-language": "zh-TW"}
    if near:
        lat, lng = near
        d = 0.25  # ~25km 視窗做鄰近偏好（不強制）
        params["viewbox"] = f"{lng - d},{lat + d},{lng + d},{lat - d}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(NOMINATIM_SEARCH_URL, params=params,
                                  headers={"User-Agent": GEO_UA}, timeout=10)
            r.raise_for_status()
            arr = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[guide] geocode failed for '{query}': {e}")
        return None
    if not arr:
        return None
    top = arr[0]
    name = (top.get("name") or top.get("display_name", query).split(",")[0]).strip()
    return float(top["lat"]), float(top["lon"]), name


class GuideTurn(BaseModel):
    role: str   # "user" | "guide"
    text: str


class GuideRequest(BaseModel):
    message: str
    lat: float | None = None
    lng: float | None = None
    # App 目前配對的相框（sync_frame 工具用；沒配對就 None）
    frame_id: int | None = None
    # 最近幾輪對話（App 帶上來，讓「第一家/那個/剛剛說的」等指代解得開）。
    # 後端只取最後 GUIDE_HISTORY_MAX 則、每則截斷，token 有天花板。
    history: list[GuideTurn] = []


GUIDE_HISTORY_MAX = 10       # 最多 5 輪往返
GUIDE_HISTORY_CHARS = 120    # 每則截斷


def _format_history(history: list[GuideTurn]) -> str:
    if not history:
        return ""
    lines = []
    for t in history[-GUIDE_HISTORY_MAX:]:
        who = "使用者" if t.role == "user" else "小憶"
        lines.append(f"{who}：{t.text[:GUIDE_HISTORY_CHARS]}")
    return "最近的對話（供理解指代，不是最新的問題）：\n" + "\n".join(lines) + "\n\n"


def build_guide_router_prompt(message: str, history: list[GuideTurn]) -> str:
    return (
        "你是旅遊 App「憶起」導遊精靈「小憶」的意圖判斷器。\n"
        + _format_history(history)
        + f"使用者最新說：{message}\n\n"
        "指代規則（嚴格遵守）：\n"
        "- 「附近／這附近／我附近／附近有…」指的是使用者「現在的 GPS 位置」，"
        "place 必須留空——即使前面聊過某個地點也一樣，不要把歷史地點當成這句的地點。\n"
        "- 只有兩種情況才填 place：(a) 這一句自己點名了地點（例：赤崁樓附近有什麼）；"
        "(b) 這一句用「那附近／那邊／那裡附近」明確指前面提過的地點。\n"
        "- 「它／那個／第一家／剛剛說的」等指代，依上面的對話解析成具體地點。\n\n"
        "判斷意圖並輸出 JSON：\n"
        "- nearby：找附近店家/景點（例：附近有冰店嗎、走路10分鐘有什麼好吃的）\n"
        "- route：問怎麼去某地、去某地要多久、某地離這裡多遠"
        "（例：從這裡去赤崁樓要多久、怎麼去火車站、林百貨多遠）\n"
        "- describe：想了解某地點的故事/介紹（例：介紹這個景點、赤崁樓的故事）\n"
        "- greeting：打招呼問候（例：你好、嗨、哈囉）\n"
        "- chat：其他閒聊\n\n"
        '只輸出 JSON：{"intent":"nearby|route|describe|greeting|chat",'
        '"category":"food 或 sight 或 any 或具體分類（冰品/咖啡廳/餐廳/古蹟/景點/公園…）",'
        '"keyword":"nearby 時使用者指定的「具體品項」（例：牛肉湯、豆花、拉麵、鹹粥）。'
        '只是泛稱（好吃的/美食/景點）就空字串",'
        '"minutes":走路分鐘數（使用者有講才填，否則 10）,'
        '"place":"「地點名」：nearby 時若使用者指定某地附近（例：赤崁樓附近有什麼吃的'
        '→ 赤崁樓）；route 的目的地；describe 的對象。沒指定＝空字串（以使用者現在位置為準）",'
        '"reply":"greeting/chat 時你溫暖口語的回覆（1-2句繁中，第一人稱小憶）；'
        'nearby/describe 留空"}'
    )


def build_guide_nearby_prompt(message: str, spots: list[dict],
                              anchor: str = "") -> str:
    where = f"「{anchor}」附近" if anchor else "使用者附近"
    listing = "\n".join(
        f"{i+1}. {s['name']}（{s['category']}）約 {int(s['distance_m'])} 公尺"
        for i, s in enumerate(spots)
    )
    return (
        "你是導遊精靈小憶，親切口語。使用者問：" + message + "\n"
        + where + f"符合的地點（由近到遠，距離是離{anchor or '使用者'}的直線距離）：\n"
        + listing + "\n\n"
        "用 2 到 3 句繁體中文、溫暖口語推薦其中幾個，可換算步行時間"
        f"（每分鐘約 {GUIDE_WALK_M_PER_MIN} 公尺）。只根據清單，不要編造。"
        '只輸出 JSON：{"reply":"..."}'
    )


def build_guide_describe_prompt(place: str) -> str:
    return (
        f"你是導遊精靈小憶。使用者想了解「{place}」。"
        "用 3 到 4 句繁體中文、溫暖口語介紹它的特色或小故事，像在地導遊。"
        "若你不確定真實的歷史細節，就描寫可以怎麼欣賞、體驗這裡，不要編造具體史實。"
        '只輸出 JSON：{"reply":"..."}'
    )


async def _guide_pool(lat: float, lng: float, radius_m: int,
                      query: str | None = None) -> list[dict]:
    return await find_pois(lat, lng, radius_m, budget_s=20, query=query)


def _guide_filter(pool: list[dict], category: str, limit: int = 6) -> list[dict]:
    if category in ("food", "sight"):
        want_food = category == "food"
        cands = [s for s in pool if (s["category"] in FOOD_KINDS) == want_food]
    elif category and category not in ("any", ""):
        cands = [s for s in pool if s["category"] == category]
    else:
        return balanced_pick(pool, limit)
    cands.sort(key=lambda s: s["distance_m"])
    return cands[:limit] or balanced_pick(pool, limit)


async def _guide_ask_classic(req: GuideRequest):
    """固定意圖分類版（router→工具→組答）。作為 agent 版的保底：
    LLM_MOCK 模式、或 agent 迴圈出任何錯時走這裡。"""
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="說點什麼吧")

    mock = os.environ.get("LLM_MOCK") == "1"

    # 1) 路由：判斷意圖
    if mock:
        low = msg.lower()
        m = re.search(r"(?:去|到|往)\s*([一-鿿A-Za-z0-9]+?)"
                      r"(?:要|怎|多遠|有多|遠嗎|近嗎|$)", msg)
        if any(w in msg for w in ("你好", "嗨", "哈囉")) or "hi" in low:
            route = {"intent": "greeting", "reply": "你好呀～我是小憶，今天想去哪走走呢？"}
        elif any(w in msg for w in ("要多久", "怎麼去", "多遠", "多近", "距離")) and m:
            route = {"intent": "route", "place": m.group(1)}
        elif any(w in msg for w in ("冰", "吃", "喝", "咖啡", "附近", "分鐘")):
            route = {"intent": "nearby", "category": "food", "minutes": 10}
        elif "介紹" in msg or "故事" in msg:
            route = {"intent": "describe", "place": ""}
        else:
            route = {"intent": "chat", "reply": "嘿嘿，我聽著呢～想散步的話，問我附近有什麼都可以喔！"}
    else:
        try:
            route = await asyncio.to_thread(
                llm_json, build_guide_router_prompt(msg, req.history))
        except HTTPException:
            route = {"intent": "chat", "reply": "小憶剛剛恍神了，再說一次好嗎？"}

    intent = route.get("intent", "chat")

    # 2) 依意圖執行工具 / 組答
    if intent == "greeting":
        return {"reply": route.get("reply") or "你好呀～我是小憶！",
                "action": "wave", "spots": []}

    # nearby 若指定了地點錨點（赤崁樓附近…）可不需定位；其餘照舊要定位
    _needs_loc = (intent in ("describe", "route")
                  or (intent == "nearby" and not str(route.get("place") or "").strip()))
    if _needs_loc and (req.lat is None or req.lng is None):
        return {"reply": "我需要知道你在哪裡才幫得上忙～請先開啟定位喔！",
                "action": "think", "spots": []}

    if intent == "route":
        place = str(route.get("place") or "").strip()
        if not place:
            return {"reply": "你想去哪裡呀？告訴我地點名字，我幫你算算多久會到！",
                    "action": "think", "spots": []}
        geo = await geocode_place(place, near=(req.lat, req.lng))
        if geo is None:
            return {"reply": f"咦，我找不到「{place}」耶，換個說法或地標再問問看？",
                    "action": "think", "spots": []}
        tlat, tlng, tname = geo
        straight = db.haversine_m(req.lat, req.lng, tlat, tlng)
        route_m = straight * GUIDE_DETOUR
        walk = max(1, round(route_m / GUIDE_WALK_M_PER_MIN))
        bike = max(1, round(route_m / GUIDE_BIKE_M_PER_MIN))
        dist_label = (f"{straight/1000:.1f} 公里" if straight >= 1000
                      else f"{round(straight)} 公尺")
        reply = (f"從這裡到{tname}直線大約 {dist_label}，"
                 f"走路約 {walk} 分鐘、騎車約 {bike} 分鐘"
                 f"（實際看路線會再多一些）。要開導航直接點下面就好！")
        spot = {"name": tname, "lat": tlat, "lng": tlng,
                "distance_m": round(straight, 1),
                "category": "目的地", "description": f"走路約 {walk} 分鐘"}
        print(f"[guide] route to {tname}: {dist_label}, walk {walk}min")
        return {"reply": reply, "action": "point", "spots": [spot]}

    if intent == "nearby":
        minutes = int(route.get("minutes") or 10)
        radius = max(200, min(minutes * GUIDE_WALK_M_PER_MIN, GUIDE_MAX_RADIUS_M))
        keyword = str(route.get("keyword") or "").strip()

        # 搜尋中心：預設是使用者位置；「赤崁樓附近有什麼」則以赤崁樓為中心
        center_lat, center_lng, anchor = req.lat, req.lng, ""
        place = str(route.get("place") or "").strip()
        if place:
            near = (req.lat, req.lng) if req.lat is not None else None
            geo = await geocode_place(place, near=near)
            if geo is None:
                return {"reply": f"咦，我找不到「{place}」耶，換個地標名字再問問看？",
                        "action": "think", "spots": []}
            center_lat, center_lng, anchor = geo

        try:
            if keyword:
                # 具體品項（牛肉湯、豆花…）：直接用文字搜尋，範圍內沒有就擴大找一次
                spots = await _guide_pool(center_lat, center_lng, radius, query=keyword)
                if not spots and radius < GUIDE_MAX_RADIUS_M:
                    spots = await _guide_pool(
                        center_lat, center_lng, GUIDE_MAX_RADIUS_M, query=keyword)
                spots = sorted(spots, key=lambda s: s["distance_m"])[:6]
            else:
                pool = await _guide_pool(center_lat, center_lng, radius)
                spots = _guide_filter(pool, str(route.get("category") or "any"))
        except HTTPException:
            return {"reply": "附近的地圖資料剛好塞車了，等一下再問我一次好嗎？",
                    "action": "think", "spots": []}
        if not spots:
            what = f"「{keyword}」" if keyword else "符合的地方"
            where = f"{anchor}附近" if anchor else "這附近"
            return {"reply": f"{where}我沒找到{what}耶，換個說法或換樣東西再問問看？",
                    "action": "think", "spots": []}
        if mock:
            reply = f"{anchor or '這'}附近走路一下就到的有 {spots[0]['name']}，還有 {spots[1]['name'] if len(spots) > 1 else '幾個好去處'} 喔，要不要去看看？"
        else:
            try:
                data = await asyncio.to_thread(
                    llm_json, build_guide_nearby_prompt(msg, spots, anchor))
                reply = str(data.get("reply", "")).strip() or "我找到幾個地方，看看下面～"
            except HTTPException:
                reply = "我找到幾個地方，看看下面這些吧～"
        return {"reply": reply, "action": "point", "spots": spots}

    if intent == "describe":
        place = str(route.get("place") or "").strip()
        if not place:
            try:
                pool = await _guide_pool(req.lat, req.lng, 400)
                sights = [s for s in pool if s["category"] not in FOOD_KINDS]
                place = (sights or pool or [{"name": ""}])[0]["name"]
            except HTTPException:
                place = ""
        if not place:
            return {"reply": "這附近我一時找不到有故事的地標，你也可以直接告訴我地點名字！",
                    "action": "think", "spots": []}
        if mock:
            reply = f"說到{place}呀，這裡很值得放慢腳步走走，感受一下在地的氛圍，說不定會有意外的小發現喔！"
        else:
            try:
                data = await asyncio.to_thread(
                    llm_json, build_guide_describe_prompt(place))
                reply = str(data.get("reply", "")).strip() or f"{place}是個值得走走的好地方喔！"
            except HTTPException:
                reply = f"{place}是個值得走走的好地方喔！"
        return {"reply": reply, "action": "talk", "spots": []}

    # chat
    return {"reply": route.get("reply") or "嘿嘿，我在聽～", "action": "talk", "spots": []}


# --- Agent 版：LLM 自己推敲意圖、決定叫哪些工具（native function calling） ----
# 不再把話塞進固定意圖；模型看得懂就做得到（例：「帶我去最近的牛肉湯」＝
# 先 search_places 再 get_route，自己串）。gemma4 與 gemini 都實測支援
# tools API。任何一步出錯 → 自動退回上面的 classic 版。

GUIDE_TOOLS = [
    {"type": "function", "function": {
        "name": "search_places",
        "description": "搜尋某個中心點附近的店家/景點。中心點預設是使用者目前位置；"
                       "使用者指定某地附近時才填 place。",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string",
                        "description": "具體品項（牛肉湯/豆花/拉麵…）。泛稱（好吃的/景點）留空"},
            "category": {"type": "string", "enum": ["food", "sight", "any"],
                         "description": "泛搜時的類別：food=吃的喝的, sight=景點古蹟, any=都要"},
            "place": {"type": "string",
                      "description": "搜尋中心的地點名（例：赤崁樓）。用使用者目前位置時留空"},
            "minutes": {"type": "integer",
                        "description": "走路幾分鐘內（預設 10）"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_route",
        "description": "從使用者目前位置到某地點的距離與步行/騎車時間。",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "目的地名稱"}},
            "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "query_my_trips",
        "description": "查使用者「自己記錄過的旅程回憶」（真實資料：日期/距離/照片數/"
                       "參加者/有無遊記）。使用者問自己去過哪、上次旅程、走了多遠、"
                       "跟誰去、回顧行程時用這個。",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "回傳最近幾趟（預設 5，最多 10）"},
            "member": {"type": "string", "description": "只看某位參加者的旅程（例：tingyu）"},
            "keyword": {"type": "string", "description": "標題關鍵字過濾"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查目前天氣與未來幾小時降雨機率（評估適不適合出門、要不要帶傘）。",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string",
                      "description": "地點名（留空＝使用者目前位置）"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "sync_frame",
        "description": "通知使用者家裡的智慧相框開始同步（把旅程照片與遊記下載到相框）。"
                       "使用者說要同步相框、把回憶推到相框時用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "respond",
        "description": "給使用者最終回覆。完成任務時「必須」呼叫這個工具收尾。",
        "parameters": {"type": "object", "properties": {
            "reply": {"type": "string", "description": "2-4 句溫暖口語的繁體中文回覆"},
            "action": {"type": "string", "enum": ["wave", "point", "talk", "think"],
                       "description": "wave=打招呼, point=推薦了地點, talk=一般回答, "
                                      "think=道歉/需要更多資訊"}},
            "required": ["reply", "action"]}}},
]

# WMO weather code → 中文（Open-Meteo 用）
_WMO = {0: "晴朗", 1: "大致晴朗", 2: "多雲時晴", 3: "陰天", 45: "起霧", 48: "霧凇",
        51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "下雨", 65: "大雨",
        66: "凍雨", 67: "凍雨", 71: "小雪", 73: "下雪", 75: "大雪", 77: "霰",
        80: "陣雨", 81: "陣雨", 82: "強陣雨", 85: "陣雪", 86: "陣雪",
        95: "雷雨", 96: "雷雨帶冰雹", 99: "強雷雨帶冰雹"}


async def _tool_query_my_trips(args: dict, frame_id: int | None = None) -> dict:
    """frame_id: asked from a photo frame — only the trips that frame shows,
    and each with a story excerpt (the frame has no screen to open a trip on,
    so 小憶 answers "那次做了什麼？" from the text)."""
    limit = max(1, min(int(args.get("limit") or 5), 10))
    member = str(args.get("member") or "").strip().lower()
    keyword = str(args.get("keyword") or "").strip()

    stmt = (select(Trip).options(selectinload(Trip.photos))
            .order_by(Trip.created_at.desc()))
    if frame_id is not None:
        stmt = stmt.where(_frame_trip_filter(frame_id))
    async with db.SessionLocal() as session:
        trips = (await session.execute(stmt)).scalars().all()
    if frame_id is not None:
        # "上一次去哪？" means the latest trip DATE: trips imported later than
        # they happened would otherwise come first (seen: a July trip created
        # after an August one was answered as "the last trip").
        trips = sorted(trips, key=lambda t: t.start_time or t.created_at,
                       reverse=True)

    total_m = sum(t.distance_m for t in trips)
    rows = []
    for t in trips:
        members = trip_members(t)
        if member and member not in [m.lower() for m in members]:
            continue
        if keyword and keyword not in (t.title or ""):
            continue
        when = t.start_time or t.created_at
        local = when.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        row = {
            "title": t.title or "未命名旅程",
            "date": f"{local.year}-{local.month:02d}-{local.day:02d}",
            "distance_m": int(t.distance_m),
            "photos": len(t.photos),
            "members": members,
            "has_story": bool(t.story_text),
        }
        if frame_id is not None and t.story_text:
            row["story"] = t.story_text[:120]
        rows.append(row)
        if len(rows) >= limit:
            break
    return {"total_trips": len(trips),
            "total_km": round(total_m / 1000, 1),
            "trips": rows}


async def _tool_get_weather(req: GuideRequest, args: dict) -> dict:
    place = str(args.get("place") or "").strip()
    lat, lng, where = req.lat, req.lng, "你的位置"
    if place:
        geo = await geocode_place(
            place, near=(req.lat, req.lng) if req.lat is not None else None)
        if geo is None:
            return {"error": f"找不到「{place}」"}
        lat, lng, where = geo
    if lat is None:
        return {"error": "使用者沒有開定位，也沒指定地點"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lng,
                        "current": "temperature_2m,apparent_temperature,"
                                   "precipitation,weather_code",
                        "hourly": "precipitation_probability",
                        "forecast_hours": 6, "timezone": "auto"},
                timeout=10)
            r.raise_for_status()
            d = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"天氣服務暫時連不上（{e}）"}
    cur = d.get("current", {})
    probs = (d.get("hourly", {}).get("precipitation_probability") or [])[:6]
    return {
        "place": where,
        "now": {
            "weather": _WMO.get(int(cur.get("weather_code", -1)), "未知"),
            "temp_c": cur.get("temperature_2m"),
            "feels_c": cur.get("apparent_temperature"),
            "precip_mm": cur.get("precipitation"),
        },
        "rain_prob_next_6h_percent": probs,
    }


async def _tool_sync_frame(req: GuideRequest) -> dict:
    if req.frame_id is None:
        return {"error": "使用者的 App 還沒配對相框，請 respond 引導他到「相框」頁配對"}
    async with db.SessionLocal() as session:
        frame = (
            await session.execute(select(Frame).where(Frame.id == req.frame_id))
        ).scalar_one_or_none()
        if frame is None:
            return {"error": "找不到這台相框，可能已被解除配對"}
        frame.sync_requested = 1
        await session.commit()
    print(f"[guide] frame {req.frame_id}: sync requested via 小憶")
    return {"ok": True, "note": "相框會在約 10 秒內開始同步照片與遊記"}


async def _tool_search_places(req: GuideRequest, args: dict, state: dict) -> dict:
    keyword = str(args.get("keyword") or "").strip()
    category = str(args.get("category") or "any").strip()
    place = str(args.get("place") or "").strip()
    minutes = int(args.get("minutes") or 10)
    radius = max(200, min(minutes * GUIDE_WALK_M_PER_MIN, GUIDE_MAX_RADIUS_M))

    lat, lng, anchor = req.lat, req.lng, ""
    if place:
        near = (req.lat, req.lng) if req.lat is not None else None
        geo = await geocode_place(place, near=near)
        if geo is None:
            return {"error": f"找不到地點「{place}」，可請使用者換個說法"}
        lat, lng, anchor = geo
    if lat is None:
        return {"error": "使用者沒有開定位，請 respond 請他開啟定位"}

    try:
        if keyword:
            spots = await _guide_pool(lat, lng, radius, query=keyword)
            if not spots and radius < GUIDE_MAX_RADIUS_M:
                spots = await _guide_pool(lat, lng, GUIDE_MAX_RADIUS_M, query=keyword)
            spots = sorted(spots, key=lambda s: s["distance_m"])[:6]
        else:
            pool = await _guide_pool(lat, lng, radius)
            spots = _guide_filter(pool, category)
    except HTTPException:
        return {"error": "地圖服務暫時塞車，請 respond 道歉並請使用者稍後再試"}

    state["spots"] = spots
    return {
        "center": anchor or "使用者目前位置",
        "results": [{
            "name": s["name"], "category": s["category"],
            "distance_m": int(s["distance_m"]),
            "walk_min": max(1, round(
                s["distance_m"] * GUIDE_DETOUR / GUIDE_WALK_M_PER_MIN)),
        } for s in spots],
    }


async def _tool_get_route(req: GuideRequest, args: dict, state: dict) -> dict:
    place = str(args.get("place") or "").strip()
    if not place:
        return {"error": "需要目的地名稱"}
    if req.lat is None:
        return {"error": "使用者沒有開定位，請 respond 請他開啟定位"}
    geo = await geocode_place(place, near=(req.lat, req.lng))
    if geo is None:
        return {"error": f"找不到「{place}」"}
    tlat, tlng, tname = geo
    straight = db.haversine_m(req.lat, req.lng, tlat, tlng)
    route_m = straight * GUIDE_DETOUR
    walk = max(1, round(route_m / GUIDE_WALK_M_PER_MIN))
    bike = max(1, round(route_m / GUIDE_BIKE_M_PER_MIN))
    state["dest"] = {"name": tname, "lat": tlat, "lng": tlng,
                     "distance_m": round(straight, 1), "category": "目的地",
                     "description": f"走路約 {walk} 分鐘"}
    return {"place": tname, "straight_m": int(straight),
            "walk_min": walk, "bike_min": bike}


def _llm_tools_step(messages: list[dict]):
    """一步 agent 呼叫（帶工具）。與 llm_json 同一套路由設定。"""
    extra = {}
    if LLM_API_BASE:
        extra["api_base"] = LLM_API_BASE
    if os.environ.get("LLM_API_KEY"):
        extra["api_key"] = os.environ["LLM_API_KEY"]
    if LLM_REASONING_EFFORT:
        extra["reasoning_effort"] = LLM_REASONING_EFFORT
        extra["allowed_openai_params"] = ["reasoning_effort"]
    resp = litellm.completion(
        model=LLM_MODEL, messages=messages,
        tools=GUIDE_TOOLS, tool_choice="auto",
        temperature=0.7, timeout=40, **extra)
    return resp.choices[0].message


def _agent_response(reply: str, action: str, state: dict) -> dict:
    spots = list(state.get("spots") or [])
    dest = state.get("dest")
    if dest:
        # 目的地卡片（有導航按鈕）永遠保留，搜尋結果讓位
        spots = [s for s in spots if s["name"] != dest["name"]][:5]
        spots.append(dest)
        return {"reply": reply, "action": action, "spots": spots}
    return {"reply": reply, "action": action, "spots": spots[:6]}


async def _guide_ask_agent(req: GuideRequest, msg: str) -> dict:
    state: dict = {"spots": [], "dest": None}
    has_loc = req.lat is not None and req.lng is not None
    system = (
        "你是旅遊 App「憶起」的導遊精靈「小憶」，個性溫暖、口語、講繁體中文。"
        "你可以呼叫工具完成使用者的請求，需要幾次就叫幾次"
        "（例：「帶我去最近的牛肉湯」先 search_places 找到店，再 get_route 算路程）。\n"
        "規則：\n"
        "1. 「附近／這附近」一律指使用者目前位置（search_places 的 place 留空）；"
        "只有使用者這句自己點名地點、或明說「那附近」才帶 place。\n"
        "2. 推薦的店名只能來自工具結果，嚴禁自己編造；查不到就誠實說。\n"
        "3. 介紹地點的故事可用你的知識，但不確定的史實不要編，可改聊怎麼欣賞。\n"
        f"4. 使用者{'有' if has_loc else '沒有'}提供定位。\n"
        "5. 完成時「必須」呼叫 respond 工具收尾（打招呼用 action=wave，"
        "推薦了地點用 point，一般回答 talk，道歉或要資訊 think）。\n"
        "6. 有效率：同一個工具不要用一樣的參數重複呼叫；工具結果夠回答了就"
        "立刻 respond，不要多繞。"
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for t in req.history[-GUIDE_HISTORY_MAX:]:
        messages.append({"role": "user" if t.role == "user" else "assistant",
                         "content": t.text[:GUIDE_HISTORY_CHARS]})
    messages.append({"role": "user", "content": msg})

    for _ in range(6):                       # 最多 6 步（含收尾）
        m = await asyncio.to_thread(_llm_tools_step, messages)
        tool_calls = getattr(m, "tool_calls", None)

        if not tool_calls:                   # 模型直接回文字：當作最終回覆
            reply = (m.content or "").strip()
            if reply:
                return _agent_response(reply, "talk", state)
            raise RuntimeError("empty agent reply")

        messages.append({
            "role": "assistant", "content": m.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            if name == "respond":
                reply = str(args.get("reply") or "").strip()
                action = str(args.get("action") or "talk")
                if action not in ("wave", "point", "talk", "think"):
                    action = "talk"
                if reply:
                    print(f"[guide] agent done: action={action}, "
                          f"spots={len(state.get('spots') or [])}, "
                          f"dest={'yes' if state.get('dest') else 'no'}")
                    return _agent_response(reply, action, state)
                result: dict = {"error": "reply 不可為空，請重新 respond"}
            elif name == "search_places":
                result = await _tool_search_places(req, args, state)
            elif name == "get_route":
                result = await _tool_get_route(req, args, state)
            elif name == "query_my_trips":
                result = await _tool_query_my_trips(args)
            elif name == "get_weather":
                result = await _tool_get_weather(req, args)
            elif name == "sync_frame":
                result = await _tool_sync_frame(req)
            else:
                result = {"error": f"沒有 {name} 這個工具"}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    raise RuntimeError("agent loop exceeded max steps")


class SpeakRequest(BaseModel):
    text: str


@app.post("/guide/speak")
async def guide_speak(req: SpeakRequest):
    """小憶開口說話：文字 → mp3（edge-tts，與相框同一個聲音 HsiaoChen）。"""
    text = req.text.strip()[:300]
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    import edge_tts

    mp3_path = AUDIO_DIR / f"speak_{uuid.uuid4().hex[:8]}.mp3"
    try:
        communicate = edge_tts.Communicate(
            text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
        await communicate.save(str(mp3_path))
        data = mp3_path.read_bytes()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
    finally:
        mp3_path.unlink(missing_ok=True)
    return Response(content=data, media_type="audio/mpeg")


@app.post("/guide/ask")
async def guide_ask(req: GuideRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="說點什麼吧")
    if os.environ.get("LLM_MOCK") == "1":
        return await _guide_ask_classic(req)
    try:
        return await _guide_ask_agent(req, msg)
    except Exception as e:  # noqa: BLE001 — agent 出任何錯都退回 classic
        print(f"[guide] agent failed ({type(e).__name__}: {str(e)[:150]}); "
              f"falling back to classic router")
        return await _guide_ask_classic(req)


# ================================================================ App: faces
# App 自拍註冊（隊友 Week-2 README 的「App 註冊」路線）：
#   App 上傳自拍＋名字 → 這裡轉成板端 photo-enroll 吃的 240x240 RGB565 LE raw
#   （格式與 scripts/selfie_to_frame.py 完全一致）→ 存 face_store/ →
#   相框同步時從 manifest 的 faces 段下載進 0:\faces\ → 重開機自動註冊
#   （韌體 RUN_PHOTO_ENROLL：跑板上偵測+embedding，完成後把檔案改名 .done）。

FACE_DIR = Path(__file__).parent / "face_store"
FACE_DIR.mkdir(exist_ok=True)

FACE_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,23}$")
FACE_FILE_RE = re.compile(r"^enroll_[A-Za-z0-9_-]+(-\d+)?\.raw$")
FACE_CAM_W = FACE_CAM_H = 240
FACE_MAX_PHOTOS = 8  # 韌體每次開機最多 enroll 8 張（PHOTO_ENROLL_MAX）


def selfie_to_raw(data: bytes, zoom: float = 1.0) -> bytes:
    """自拍 → 240x240 RGB565 little-endian（scripts/selfie_to_frame.py 的移植，
    輸出位元組必須完全一致：板端把它當一張相機幀跑偵測+embedding）。"""
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # 尊重手機拍攝方向
    img = img.convert("RGB")

    side = int(min(img.size) / max(zoom, 1.0))
    cx, cy = img.width // 2, img.height // 2
    img = img.crop((cx - side // 2, cy - side // 2,
                    cx - side // 2 + side, cy - side // 2 + side))
    img = img.resize((FACE_CAM_W, FACE_CAM_H), Image.LANCZOS)

    buf = bytearray(FACE_CAM_W * FACE_CAM_H * 2)
    px = img.load()
    i = 0
    for y in range(FACE_CAM_H):
        for x in range(FACE_CAM_W):
            r, g, b = px[x, y]
            v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)  # RGB565
            buf[i] = v & 0xFF                                  # little-endian
            buf[i + 1] = v >> 8
            i += 2
    return bytes(buf)


def _face_label_files(label: str) -> list[Path]:
    return sorted(
        p for p in FACE_DIR.glob(f"enroll_{label}*.raw")
        if p.name == f"enroll_{label}.raw"
        or re.fullmatch(rf"enroll_{re.escape(label)}-\d+\.raw", p.name)
    )


@app.post("/faces/enroll")
async def enroll_face(
    label: str = Form(...),
    zoom: float = Form(1.0),
    files: list[UploadFile] = File(...),
):
    label = label.strip()
    if not FACE_LABEL_RE.match(label):
        raise HTTPException(
            status_code=400,
            detail="名字只能用英數字、底線、連字號（1-23 字元），"
            "且要跟旅程參加者用同一個名字",
        )
    if not files or len(files) > FACE_MAX_PHOTOS:
        raise HTTPException(
            status_code=400, detail=f"一次 1 到 {FACE_MAX_PHOTOS} 張自拍"
        )

    for old in _face_label_files(label):  # 重新註冊 = 整組換新
        old.unlink(missing_ok=True)

    written = []
    for i, f in enumerate(files, start=1):
        data = await f.read()
        if not data:
            continue
        try:
            raw = await asyncio.to_thread(selfie_to_raw, data, zoom)
        except Exception as e:  # noqa: BLE001 — 壞圖檔
            raise HTTPException(status_code=400, detail=f"照片解析失敗：{e}")
        suffix = "" if i == 1 else f"-{i}"
        p = FACE_DIR / f"enroll_{label}{suffix}.raw"
        p.write_bytes(raw)
        written.append(p.name)

    print(f"[faces] enrolled '{label}': {len(written)} photo(s)")
    return {"label": label, "files": written}


@app.get("/faces")
async def list_faces():
    labels: dict[str, int] = {}
    for p in sorted(FACE_DIR.glob("enroll_*.raw")):
        if not FACE_FILE_RE.match(p.name):
            continue
        base = re.sub(r"-\d+$", "", p.stem[len("enroll_"):])
        labels[base] = labels.get(base, 0) + 1
    return {"faces": [
        {"label": k, "photo_count": v} for k, v in sorted(labels.items())
    ]}


@app.delete("/faces/{label}")
async def delete_face(label: str):
    if not FACE_LABEL_RE.match(label):
        raise HTTPException(status_code=400, detail="invalid label")
    removed = _face_label_files(label)
    if not removed:
        raise HTTPException(status_code=404, detail=f"未註冊：{label}")
    for p in removed:
        p.unlink(missing_ok=True)
    return {"deleted": label, "files": len(removed)}


@app.get("/faces/file/{name}")
async def get_face_file(name: str):
    """M55M1 同步下載用：faces manifest 裡的 raw 檔。"""
    if not FACE_FILE_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid face file name")
    p = FACE_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"unknown face file {name}")
    return FileResponse(p, media_type="application/octet-stream")


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


def _frame_trip_filter(frame_id: int):
    """這台相框看得到的旅程：指定給它的 + 未指定的（舊資料相容）。"""
    return (Trip.frame_id == frame_id) | (Trip.frame_id.is_(None))


async def _trip_count(session, frame_id: int) -> int:
    return len((await session.execute(
        select(Trip.id).where(_frame_trip_filter(frame_id))
    )).scalars().all())


class RegisterRequest(BaseModel):
    device_uid: str


@app.post("/frames/register")
async def register_frame(req: RegisterRequest):
    """板子開機自報身分（ESP MAC）。同一台永遠拿回同一筆 frame；
    新板子建檔並發一組唯一的 6 位配對碼（板子把它顯示在 LCD 上）。"""
    uid = req.device_uid.strip().lower()
    if not uid or len(uid) > 32:
        raise HTTPException(status_code=400, detail="bad device_uid")
    async with db.SessionLocal() as session:
        frame = (
            await session.execute(
                select(Frame).where(Frame.device_uid == uid))
        ).scalar_one_or_none()
        if frame is None:
            for _ in range(20):  # 6 位數字、避開既有碼（含 demo 的 123456）
                code = f"{random.randint(0, 999999):06d}"
                clash = (await session.execute(
                    select(Frame.id).where(Frame.pair_code == code)
                )).scalar_one_or_none()
                if clash is None:
                    break
            else:
                raise HTTPException(status_code=500, detail="pair code space busy")
            frame = Frame(pair_code=code, device_uid=uid)
            session.add(frame)
            await session.commit()
            print(f"[frames] registered new frame {frame.id} "
                  f"(uid={uid}, code={code})")
        return {
            "frame_id": frame.id,
            "pair_code": frame.pair_code,
            "name": frame.name,
        }


@app.post("/frames/{frame_id}/request-sync")
async def request_sync(frame_id: int):
    """App 的「立即同步」：立旗，板子下次問 /pending 就會來拉 /sync。"""
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        frame.sync_requested = 1
        await session.commit()
        print(f"[frames] frame {frame_id}: sync requested")
        return {"requested": True}


@app.get("/frames/{frame_id}/pending")
async def sync_pending(frame_id: int):
    """板子的門鈴輪詢（超小回應）；旗子由 /sync 拉取時清除。"""
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        return {"pending": bool(frame.sync_requested)}


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
        return frame_json(frame, await _trip_count(session, frame.id))


def _trip_version(trip: Trip) -> str:
    """一趟旅程的同步版本號。鹽值 v4：相簿多了 STORY.ULW（小憶念遊記用的
    音檔），既有相簿的版本號不會因此改變、就不會去抓新檔案——升鹽值讓板子
    整批重抓一次。（v3 是 TIM 快取 key 修正。）"""
    return hashlib.md5(
        ("v4|" + trip.story_text + "|" + trip.title + "|"
         + ",".join(str(ph.id) for ph in trip.photos) + "|"
         + ",".join(trip_members(trip))).encode("utf-8")
    ).hexdigest()[:8]


async def _frame_pending_count(session, frame: Frame) -> int:
    """「待同步」= 版本跟上次發給板子的快照不一樣的旅程（含新旅程），
    加上快照裡有、現在已不存在的旅程（App 刪除 → 板子要撤下相簿）。"""
    trips = (
        (await session.execute(
            select(Trip)
            .where(_frame_trip_filter(frame.id))
            .options(selectinload(Trip.photos))
            .order_by(Trip.created_at.desc())
        )).scalars().all()
    )
    with_content = [t for t in trips if t.story_text or t.photos]
    try:
        synced = json.loads(frame.synced_versions or "{}")
    except ValueError:
        synced = {}
    current = {
        f"T{t.id:04d}": _trip_version(t)
        for t in with_content[:FRAME_MAX_ALBUMS]
    }
    pending = sum(
        1 for folder, ver in current.items()
        if synced.get(folder) != ver
    )
    pending += sum(1 for folder in synced if folder not in current)
    return pending


@app.get("/frames/{frame_id}")
async def frame_status(frame_id: int):
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        data = frame_json(frame, await _trip_count(session, frame_id))
        data["pending_count"] = await _frame_pending_count(session, frame)
        return data


# 板端 Slideshow 的相簿上限：LIB_MAX_ALBUMS(16) 含根目錄散照的 pseudo-album，
# 所以旅程相簿最多 15 本，超過的取最新的。
FRAME_MAX_ALBUMS = 15


@app.get("/frames/{frame_id}/sync")
async def frame_sync(frame_id: int):
    """M55M1 同步清單——SD 卡結構對齊板端 Slideshow 的相簿架構：
      0:\\pictures\\<folder>\\  = 一趟旅程一本相簿
        Pxxxx.JPG   照片（board 尺寸 baseline JPEG）
        LABEL.JSON  參加者 {"users": [...]}（人臉辨識過濾用）
        STORY.TXT / STORY.TIM / STORY.WAV  遊記（Slideshow 掃圖時自動略過）
        VERSION.TXT 韌體自己寫，跟 manifest 的 version 比對，相同→整趟跳過
    """
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        trips = (
            (await session.execute(
                select(Trip)
                .where(_frame_trip_filter(frame_id))
                .options(selectinload(Trip.photos))
                .order_by(Trip.created_at.desc())
            )).scalars().all()
        )

        with_content = [t for t in trips if t.story_text or t.photos]
        truncated = len(with_content) > FRAME_MAX_ALBUMS
        selected = list(reversed(with_content[:FRAME_MAX_ALBUMS]))  # 舊→新排列

        items = []
        for trip in selected:
            date_str = ""
            if trip.start_time:
                local = trip.start_time.replace(
                    tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
                date_str = f"{local.year}-{local.month:02d}-{local.day:02d}"
            members = trip_members(trip)
            version = _trip_version(trip)
            has_story = bool(trip.story_text)
            items.append({
                "trip_id": trip.id,
                "folder": f"T{trip.id:04d}",
                "title": trip.title or "未命名旅程",
                "date": date_str,
                "version": version,
                "users": members,
                "label_json": f"/trips/{trip.id}/label.json",
                "has_story": has_story,
                "story_txt": f"/trips/{trip.id}/story.txt" if has_story else None,
                "story_tim": f"/trips/{trip.id}/story.tim" if has_story else None,
                "story_wav": f"/trips/{trip.id}/story.wav" if has_story else None,
                # 相框念遊記用的音檔（8 kHz mu-law，見 /trips/{id}/story.ulaw）
                "story_ulaw": f"/trips/{trip.id}/story.ulaw" if has_story else None,
                "photos": [
                    {"name": f"P{ph.id:04d}.JPG", "url": f"/photos/{ph.id}/board"}
                    for ph in trip.photos
                ],
            })

        # 自拍註冊檔 → 0:\faces\。version=內容 hash：韌體端若同名 .done 的
        # version 相同就跳過（避免每次同步都重複註冊）。
        faces = []
        for p in sorted(FACE_DIR.glob("enroll_*.raw")):
            if not FACE_FILE_RE.match(p.name):
                continue
            faces.append({
                "name": p.name,
                "url": f"/faces/file/{p.name}",
                "size": p.stat().st_size,
                "version": hashlib.md5(p.read_bytes()).hexdigest()[:8],
            })

        frame.last_sync = db.utcnow()
        frame.sync_requested = 0            # 門鈴旗：拉取即消化
        # 記下這次發出去的版本快照 → /frames/{id} 算「待同步」數量用。
        frame.synced_versions = json.dumps(
            {it["folder"]: it["version"] for it in items})
        await session.commit()
        if truncated:
            print(f"[frames] sync truncated to newest {FRAME_MAX_ALBUMS} "
                  f"of {len(with_content)} trips (board album cap)")
        print(f"[frames] frame {frame_id} sync manifest: "
              f"{len(items)} trip(s), {len(faces)} face file(s)")
        return {
            "sd_root": "pictures",
            "faces_root": "faces",
            "trips": items,
            "faces": faces,
            "count": len(items),
            "truncated": truncated,
        }


# ---------------------------------------------------------------------------
# 手勢按讚：板子看到長輩對相框比讚 → POST；App 輪詢新事件跳通知。
# ---------------------------------------------------------------------------

class LikeIn(BaseModel):
    folder: str = ""     # 相框當下播的相簿，例 "T0006"（"" = 根目錄散照）
    photo: str = ""      # 當下照片檔名，例 "P0012.JPG"（"" = 純遊記畫面）
    user: str = ""       # 人臉辨識 label（"" = 沒認出）


def _like_ids(folder: str, photo: str) -> tuple[int | None, int | None]:
    """Board names -> (trip_id, photo_id): "T0006" -> 6, "P0012.JPG" -> 12."""
    trip_id = photo_id = None
    m = re.fullmatch(r"T(\d{1,8})", folder.strip())
    if m:
        trip_id = int(m.group(1))
    m = re.fullmatch(r"P(\d{1,8})\.JPG", photo.strip(), re.IGNORECASE)
    if m:
        photo_id = int(m.group(1))
    return trip_id, photo_id


async def _record_like(session, frame_id: int, folder: str, photo: str,
                       user: str) -> LikeEvent:
    """One like, however it was given (thumbs-up gesture or said to 小憶);
    the App polls /likes and notifies the family either way."""
    trip_id, photo_id = _like_ids(folder, photo)
    ev = LikeEvent(
        frame_id=frame_id,
        trip_id=trip_id,
        photo_id=photo_id,
        user_label=user.strip()[:64],
    )
    session.add(ev)
    await session.commit()
    print(f"[like] frame {frame_id}: '{user or '?'}' liked "
          f"{folder or '-'}/{photo or '-'}")
    return ev


@app.post("/frames/{frame_id}/like")
async def frame_like(frame_id: int, body: LikeIn):
    async with db.SessionLocal() as session:
        await get_frame_or_404(session, frame_id)
        ev = await _record_like(session, frame_id, body.folder, body.photo,
                                body.user)
        return {"ok": True, "like_id": ev.id}


@app.get("/frames/{frame_id}/likes")
async def frame_likes(frame_id: int, after: int = 0, limit: int = 20):
    """App 輪詢用：回傳 id > after 的按讚事件（舊到新）。"""
    async with db.SessionLocal() as session:
        rows = (
            (await session.execute(
                select(LikeEvent)
                .where(LikeEvent.frame_id == frame_id, LikeEvent.id > after)
                .order_by(LikeEvent.id.asc())
                .limit(max(1, min(limit, 50)))
            )).scalars().all()
        )
        likes = []
        for ev in rows:
            title = None
            if ev.trip_id is not None:
                trip = await session.get(Trip, ev.trip_id)
                title = (trip.title or "未命名旅程") if trip else None
            likes.append({
                "id": ev.id,
                "trip_id": ev.trip_id,
                "photo_id": ev.photo_id,
                "trip_title": title,
                "user": ev.user_label,
                "created_at": db.iso_z(ev.created_at),
            })
        return {"likes": likes,
                "last_id": likes[-1]["id"] if likes else after}


# ---------------------------------------------------------------------------
# 長輩留言：對相框說「跟女兒說…」→ 小憶的 leave_message 工具存進
# frame_messages；App 輪詢新留言跳通知，也能聽長輩講這句話的原音。
# ---------------------------------------------------------------------------

@app.get("/frames/{frame_id}/messages")
async def frame_messages(frame_id: int, after: int = 0, limit: int = 20,
                         newest: bool = False):
    """App 輪詢用：id > after 的留言（舊到新）。newest=true 改回傳最近
    limit 則（新到舊），給留言列表用。"""
    limit = max(1, min(limit, 50))
    async with db.SessionLocal() as session:
        stmt = select(FrameMessage).where(FrameMessage.frame_id == frame_id)
        if newest:
            stmt = stmt.order_by(FrameMessage.id.desc())
        else:
            stmt = stmt.where(FrameMessage.id > after).order_by(FrameMessage.id.asc())
        rows = (await session.execute(stmt.limit(limit))).scalars().all()

        messages = []
        for m in rows:
            title = None
            if m.trip_id is not None:
                trip = await session.get(Trip, m.trip_id)
                title = (trip.title or "未命名旅程") if trip else None
            messages.append({
                "id": m.id,
                "user": m.user_label,
                "to": m.recipient,
                "text": m.text,
                "has_audio": bool(m.voice_id)
                and (AUDIO_DIR / f"voice_{m.voice_id}.wav").exists(),
                "trip_id": m.trip_id,
                "photo_id": m.photo_id,
                "trip_title": title,
                "created_at": db.iso_z(m.created_at),
            })
        last_id = max((m["id"] for m in messages), default=after)
        return {"messages": messages, "last_id": last_id}


@app.get("/trips/{trip_id}/messages")
async def trip_messages(trip_id: int, limit: int = 20):
    """這趟旅程的相框留言（新到舊）：長輩看著這本相簿留的話，就掛在這趟
    的詳情頁上。原音端點要 frame_id，所以一起回傳。"""
    limit = max(1, min(limit, 50))
    async with db.SessionLocal() as session:
        rows = (await session.execute(
            select(FrameMessage)
            .where(FrameMessage.trip_id == trip_id)
            .order_by(FrameMessage.id.desc())
            .limit(limit)
        )).scalars().all()
    return {"messages": [{
        "id": m.id,
        "frame_id": m.frame_id,
        "user": m.user_label,
        "to": m.recipient,
        "text": m.text,
        "has_audio": bool(m.voice_id)
        and (AUDIO_DIR / f"voice_{m.voice_id}.wav").exists(),
        "photo_id": m.photo_id,
        "created_at": db.iso_z(m.created_at),
    } for m in rows]}


@app.get("/frames/{frame_id}/messages/{message_id}/audio")
async def frame_message_audio(frame_id: int, message_id: int):
    """長輩講這則留言的原音，轉成手機好播的 MP3。

    相框錄的 WAV 不能直接給：取樣率是 DMIC 除得出來的 ~8263 Hz（有些手機
    播放器不收），而且板載麥克風很小聲（實測平均 -57 dB、峰值 -28 dB）。
    這裡高通去低頻雜訊 + EBU R128 音量正規化 + 16 kHz MP3，快取一份。"""
    async with db.SessionLocal() as session:
        m = await session.get(FrameMessage, message_id)
    if m is None or m.frame_id != frame_id or not m.voice_id:
        raise HTTPException(status_code=404, detail="unknown message")
    wav = AUDIO_DIR / f"voice_{m.voice_id}.wav"
    mp3 = AUDIO_DIR / f"voice_{m.voice_id}.mp3"
    if not mp3.exists():
        if not wav.exists():
            raise HTTPException(status_code=404, detail="recording no longer kept")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(wav),
            "-af", "highpass=f=120,loudnorm=I=-18:TP=-2:LRA=11",
            "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "48k",
            str(mp3),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            mp3.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"transcode failed: {stderr.decode(errors='replace')[-300:]}")
    return FileResponse(mp3, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# 語音問小憶：長輩按住相框 BTN0 講話 → 板子錄 8 kHz μ-law → POST 到這裡。
# 聽寫和回答是同一次 LLM 呼叫（Gemini 直接吃音訊），附上相框當下的情境
# （播到哪本相簿、哪張照片、認出誰）與那張照片本身；回覆畫成故事欄的
# TIM4 直接回傳，板子一次往返就能上畫面。
# ---------------------------------------------------------------------------

VOICE_MAX_BYTES = 16 * 8000     # 16 s of 8 kHz mu-law, far past what the board sends
VOICE_HISTORY_TURNS = 3         # follow-ups ("那是什麼時候？") need the last few turns
VOICE_HISTORY_TTL_S = 180       # three quiet minutes = a new conversation

# frame_id -> {"t": last activity, "turns": [(question, reply), ...]}
VOICE_HISTORY: dict[int, dict] = {}

# The rail font (Microsoft JhengHei) has no emoji: they would render as boxes.
_NO_GLYPH_RE = re.compile("[\U00010000-\U0010FFFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")

VOICE_PERSONA = (
    "你是智慧相框「憶起」裡的精靈「小憶」，陪家裡的長輩看家人旅遊的照片。"
    "個性溫暖、口語、有耐心，講繁體中文。\n"
    "使用者剛按著相框的按鈕對你說了一段話（附上錄音）。"
    "先把他說的話聽寫下來，再回答他。\n"
    "規則：\n"
    "1. 回答要短：最多 45 個字、一到兩句——會顯示在相框側邊的窄欄上。\n"
    "2. 事實（地點、日期、人物）只能根據相框資料、照片畫面和工具查到的資料；"
    "都沒有的就老實說不確定，可以改聊照片裡看得到的東西。\n"
    "3. 錄音裡沒人說話或聽不清楚時，transcript 給空字串，reply 請他再說一次。\n"
    "4. 不要表情符號、不要引號、不要條列。\n"
)

# Fallback when the tool loop fails: one plain call, JSON out.
VOICE_SYSTEM_JSON = (
    VOICE_PERSONA
    + '只輸出 JSON：{"transcript": "聽到的原話（繁體中文）", "reply": "小憶的回答"}'
)

VOICE_SYSTEM_TOOLS = (
    VOICE_PERSONA
    + "5. 問眼前這張照片或這本相簿的事，相框資料已經夠了，直接 respond，"
    "不要呼叫查詢工具（每多呼叫一次，長輩就要多等好幾秒）。"
    "只有問題超出這本相簿時（去過哪些地方、上次旅程、跟誰去…）才呼叫工具查；"
    "同一個工具不要用一樣的參數重複呼叫。\n"
    "6. 完成時「必須」呼叫 respond 收尾：transcript 填他這次說的原話，"
    "reply 填你的回答。\n"
    "7. 只能說你「真的做了」的事。按讚、換相簿、換照片、留言給家人、念遊記，"
    "都要先呼叫對應的工具才算數；沒呼叫工具，就不要在回覆裡說你做了或會去做。"
    "長輩說「我喜歡這張」就是要按讚，要按就呼叫 like_photo，"
    "不按就不要說你會告訴家人。\n"
)

# 相框版小憶的工具（App 版見 GUIDE_TOOLS）。一個一個加：每個工具在
# _voice_agent 的分派裡有對應的一支 _tool_*。
VOICE_TOOLS = [
    {"type": "function", "function": {
        "name": "query_my_trips",
        "description": "查這台相框裡家人的旅程回憶（真實資料：日期、距離、照片數、"
                       "參加者、遊記摘要）。問到去過哪裡、上次旅程、跟誰去、"
                       "走了多遠等「不只眼前這張照片」的問題時用。",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "回傳最近幾趟（預設 5，最多 10）"},
            "member": {"type": "string",
                       "description": "只看某位參加者的旅程（人臉辨識的名字，例：user1）"},
            "keyword": {"type": "string", "description": "標題關鍵字過濾（例：日月潭）"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "sync_frame",
        "description": "檢查家人有沒有新的旅程照片或遊記，並讓相框馬上下載更新。"
                       "問有沒有新照片、要更新或同步相框時用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "like_photo",
        "description": "幫長輩對「眼前這張照片」按讚，家人的 App 會收到通知（跟對相框比讚一樣）。"
                       "他說喜歡這張、要按讚、想讓家人知道他喜歡時用；"
                       "只是在聊照片內容就不要按。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "show_album",
        "description": "讓相框馬上換到某一本相簿播放（例：給我看台北的照片、我想看日月潭那次）。"
                       "用地點／標題／景點關鍵字或參加者找相簿，不用先查旅程；"
                       "找不到會回報相框裡有哪些相簿。",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string",
                        "description": "地點、相簿標題或景點的關鍵字，只填詞本身（例：台北、日月潭）"},
            "member": {"type": "string",
                       "description": "想看誰的照片（人臉辨識的名字，例：user1）"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "control_slideshow",
        "description": "控制相框輪播：下一張、上一張、停在這張（暫停自動換）、繼續播。",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "enum": ["next", "previous", "pause", "resume"],
                        "description": "next=下一張, previous=上一張, "
                                       "pause=停在這張, resume=繼續播"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_story",
        "description": "把這本相簿的遊記用相框的喇叭念給長輩聽（念的是遊記全文，不是你的回覆）。"
                       "他說念給我聽、唸一下這趟的故事、字太小看不清楚時用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "leave_message",
        "description": "幫長輩留話給家人（例：跟女兒說我很想她、告訴大家週末回來吃飯）。"
                       "家人的 App 會跳通知，也能聽到長輩講這句話的原音。",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string",
                   "description": "給誰，用長輩的稱呼（例：女兒、孫子、全家）；沒說就填「家人」"},
            "message": {"type": "string",
                        "description": "要轉達的話，用長輩的口吻、第一人稱，"
                                       "保留他的意思不要加油添醋（例：我很想妳，有空回來吃飯）"}},
            "required": ["to", "message"]}}},
    {"type": "function", "function": {
        "name": "respond",
        "description": "給長輩最終回覆並收尾。每次都「必須」呼叫這個工具結束。",
        "parameters": {"type": "object", "properties": {
            "transcript": {"type": "string",
                           "description": "他這次錄音說的原話（繁體中文聽寫）；"
                                          "沒人說話或聽不清楚給空字串"},
            "reply": {"type": "string",
                      "description": "小憶的回答，最多 45 個字、一到兩句"}},
            "required": ["transcript", "reply"]}}},
]
VOICE_AGENT_STEPS = 4           # tool rounds + the respond; each is one LLM call


def _ulaw_decode_table() -> list[int]:
    """G.711 mu-law byte -> PCM16, the inverse of the board's mulaw_encode."""
    table = []
    for b in range(256):
        u = ~b & 0xFF
        mag = ((((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 0x07)) - 0x84
        table.append(-mag if u & 0x80 else mag)
    return table


_ULAW_TO_PCM = _ulaw_decode_table()


def ulaw_to_wav(data: bytes, rate: int) -> bytes:
    """8-bit mu-law -> 16-bit mono WAV at the rate the board really sampled
    (its DMIC runs off APLL1, so e.g. 8263 Hz rather than 8000)."""
    import io
    import wave
    from array import array

    pcm = array("h", (_ULAW_TO_PCM[b] for b in data))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


async def _voice_context(session, folder: str, photo: str,
                         user: str) -> tuple[str, Path | None]:
    """What the frame is showing: prompt lines + the photo file on screen."""
    lines: list[str] = []
    image_path = None

    m = re.fullmatch(r"T(\d{1,8})", folder.strip())
    trip = await session.get(Trip, int(m.group(1))) if m else None
    if trip is not None:
        lines.append(f"正在播的相簿：{trip.title or '未命名旅程'}")
        if trip.start_time:
            local = trip.start_time.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
            lines.append(f"旅程日期：{local.year}年{local.month}月{local.day}日")
        members = trip_members(trip)
        if members:
            lines.append("同行的人：" + "、".join(members))
        spots = (await session.execute(
            select(TripSpot.name).where(TripSpot.trip_id == trip.id)
        )).scalars().all()
        if spots:
            lines.append("這趟收藏的景點：" + "、".join(spots[:8]))
        if trip.story_text:
            lines.append(f"這趟的遊記：{trip.story_text}")

    m = re.fullmatch(r"P(\d{1,8})\.JPG", photo.strip(), re.IGNORECASE)
    ph = await session.get(Photo, int(m.group(1))) if m else None
    if ph is not None:
        if ph.timestamp:
            local = ph.timestamp.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
            lines.append(f"螢幕上這張照片拍攝於：{local.year}年{local.month}月"
                         f"{local.day}日 {local:%H:%M}")
        path = PHOTO_DIR / ph.filename
        if path.exists():
            image_path = path
            lines.append("螢幕上這張照片附在後面。")
    elif trip is not None and not photo.strip():
        lines.append("螢幕上現在是這趟的遊記畫面，不是照片。")

    if user.strip():
        lines.append(f"人臉辨識認出正在看相框的是：{user.strip()}")
    if not lines:
        lines.append("（相框沒有提供目前畫面的資料）")
    return "\n".join(lines), image_path


def _voice_turns(frame_id: int) -> list[tuple[str, str]]:
    h = VOICE_HISTORY.get(frame_id)
    if not h or time.time() - h["t"] > VOICE_HISTORY_TTL_S:
        return []
    return list(h["turns"])


def _voice_remember(frame_id: int, question: str, reply: str) -> None:
    turns = _voice_turns(frame_id) + [(question, reply)]
    VOICE_HISTORY[frame_id] = {"t": time.time(),
                               "turns": turns[-VOICE_HISTORY_TURNS:]}


def _voice_llm_extra() -> dict:
    """Routing kwargs for litellm — same env as llm_json / the App guide."""
    key_vars = required_api_keys(LLM_MODEL)
    if not any(os.environ.get(k) for k in key_vars):
        raise RuntimeError(f"none of {key_vars} is set for model {LLM_MODEL}")
    extra = {}
    if LLM_API_BASE:
        extra["api_base"] = LLM_API_BASE
    if os.environ.get("LLM_API_KEY"):
        extra["api_key"] = os.environ["LLM_API_KEY"]
    if LLM_REASONING_EFFORT:
        extra["reasoning_effort"] = LLM_REASONING_EFFORT
        extra["allowed_openai_params"] = ["reasoning_effort"]
    return extra


def _voice_messages(system: str, context: str, image_uri: str | None,
                    wav: bytes, turns: list[tuple[str, str]]) -> list[dict]:
    """System prompt, the last few turns as text, then this question: frame
    context + the photo on screen + the recording."""
    import base64

    messages: list[dict] = [{"role": "system", "content": system}]
    for q, a in turns:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    content: list[dict] = [{"type": "text", "text": "相框資料：\n" + context}]
    if image_uri:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    content.append({"type": "input_audio", "input_audio": {
        "data": base64.b64encode(wav).decode(), "format": "wav"}})
    messages.append({"role": "user", "content": content})
    return messages


def _voice_llm_step(messages: list[dict]):
    """One agent step (tools offered). Returns the assistant message."""
    resp = litellm.completion(
        model=LLM_MODEL,
        messages=messages,
        tools=VOICE_TOOLS,
        tool_choice="auto",
        temperature=0.7,
        timeout=40,
        **_voice_llm_extra(),
    )
    return resp.choices[0].message


async def _voice_tool_sync(frame_id: int, state: dict) -> dict:
    """Doorbell up (as the App's 「立即同步」) AND tell the board right away via
    X-Frame-Action, so it syncs as soon as 小憶 has answered instead of after
    the reply hold plus a doorbell interval. The flag alone still works for a
    board that ignores the header."""
    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        pending = await _frame_pending_count(session, frame)
        frame.sync_requested = 1
        await session.commit()
    state["action"] = "sync"
    print(f"[voice] frame {frame_id}: sync requested ({pending} album(s) pending)")
    return {"ok": True, "new_or_changed_albums": pending,
            "note": ("相框回答完就會開始下載" if pending
                     else "目前沒有新的相簿，相框還是會再檢查一次")}


VOICE_LIKE_DEDUP_S = 300       # "我喜歡這張" said twice = one like, one notification


async def _voice_tool_like(frame_id: int, screen: dict) -> dict:
    """Like the photo on screen — the same record the thumbs-up gesture makes,
    so the family's App notifies the same way. A repeat by the same viewer
    within VOICE_LIKE_DEDUP_S is not recorded again."""
    folder, photo, user = screen["folder"], screen["photo"], screen["user"]
    trip_id, photo_id = _like_ids(folder, photo)
    if trip_id is None and photo_id is None:
        return {"error": "相框現在播的不是家人的旅程相簿，沒辦法按讚"}

    since = db.utcnow() - timedelta(seconds=VOICE_LIKE_DEDUP_S)
    async with db.SessionLocal() as session:
        dup = (await session.execute(
            select(LikeEvent.id).where(
                LikeEvent.frame_id == frame_id,
                LikeEvent.trip_id.is_(None) if trip_id is None
                else LikeEvent.trip_id == trip_id,
                LikeEvent.photo_id.is_(None) if photo_id is None
                else LikeEvent.photo_id == photo_id,
                LikeEvent.user_label == user.strip()[:64],
                LikeEvent.created_at >= since,
            ).limit(1)
        )).scalar_one_or_none()
        if dup is not None:
            return {"ok": True, "already_liked": True,
                    "note": "剛剛已經按過讚了，家人已經收到"}
        await _record_like(session, frame_id, folder, photo, user)
    return {"ok": True, "note": "已按讚，家人的 App 會收到通知"}


async def _voice_tool_show_album(frame_id: int, args: dict, state: dict) -> dict:
    """Pick one album the card really has — the folders of the last sync
    manifest — by keyword (title, story or saved spot) and/or member, newest
    trip first. The board jumps to it (X-Frame-Action: album=T0016)."""
    keyword = str(args.get("keyword") or "").strip()
    member = str(args.get("member") or "").strip().lower()
    if not keyword and not member:
        return {"error": "需要相簿的地點／標題關鍵字或人名"}

    async with db.SessionLocal() as session:
        frame = await get_frame_or_404(session, frame_id)
        try:
            on_card = set(json.loads(frame.synced_versions or "{}"))
        except ValueError:
            on_card = set()
        trips = (await session.execute(
            select(Trip).where(_frame_trip_filter(frame_id))
            .options(selectinload(Trip.photos))
        )).scalars().all()
        spots: dict[int, list[str]] = {}
        for tid, name in (await session.execute(
            select(TripSpot.trip_id, TripSpot.name)
            .where(TripSpot.trip_id.in_([t.id for t in trips]))
        )).all():
            spots.setdefault(tid, []).append(name)

    def matches(t: Trip) -> bool:
        if member and member not in [m.lower() for m in trip_members(t)]:
            return False
        return (not keyword
                or keyword in (t.title or "")
                or keyword in (t.story_text or "")
                or any(keyword in s for s in spots.get(t.id, [])))

    found = sorted((t for t in trips if matches(t)),
                   key=lambda t: t.start_time or t.created_at, reverse=True)
    playable = [t for t in found if f"T{t.id:04d}" in on_card]
    if not playable:
        if found:
            return {"error": f"找到「{found[0].title or '未命名旅程'}」，但還沒同步到相框"
                             "（可以問他要不要更新相框）"}
        return {"error": "相框裡沒有符合的相簿",
                "albums_on_frame": [t.title or "未命名旅程" for t in trips
                                    if f"T{t.id:04d}" in on_card]}

    t = playable[0]
    state["action"] = f"album=T{t.id:04d}"
    when = (t.start_time or t.created_at).replace(
        tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
    return {"ok": True, "album": t.title or "未命名旅程",
            "date": f"{when.year}-{when.month:02d}-{when.day:02d}",
            "photos": len(t.photos),
            "other_matches": [x.title or "未命名旅程" for x in playable[1:4]]}


# control_slideshow command -> board token after "photo=" (VoiceChat.h)
_SLIDESHOW_CMDS = {"next": "next", "previous": "prev",
                   "pause": "pause", "resume": "resume"}
_SLIDESHOW_NOTES = {
    # 小憶 cannot see the photo the board switches to: keep her from
    # describing it from the old one.
    "next": "相框換到下一張了；新照片你看不到，不要描述它",
    "prev": "相框換回上一張了（最多往回 8 張）；那張照片你看不到，不要描述它",
    "pause": "相框會停在這張，說「繼續播」就會繼續（放著十分鐘也會自己繼續）；"
             "停著的時候說下一張、上一張可以自己翻",
    "resume": "相框會繼續自動換照片",
}


async def _voice_tool_read_story(screen: dict, state: dict) -> dict:
    """Have the board read the album's story out of its phone jack. The audio
    (STORY.ULW) rides along with the album at sync time, so an album the frame
    has not synced since this feature landed has no file to play — the board
    says so on the rail."""
    trip_id, _ = _like_ids(screen["folder"], screen["photo"])
    if trip_id is None:
        return {"error": "相框現在播的不是家人的旅程相簿，沒有遊記可以念"}
    async with db.SessionLocal() as session:
        trip = await session.get(Trip, trip_id)
    if trip is None or not trip.story_text:
        return {"error": "這本相簿還沒有遊記"}
    state["action"] = "story=read"
    return {"ok": True, "album": trip.title or "未命名旅程",
            "note": "相框馬上念出來；回覆一句就好，不要把遊記唸進回覆裡"}


async def _voice_tool_message(frame_id: int, args: dict, screen: dict) -> dict:
    """Store a message for the family, linked to this recording so the App can
    play the elder's own voice, and to the photo that was on screen."""
    text = str(args.get("message") or "").strip()[:300]
    to = str(args.get("to") or "").strip()[:64] or "家人"
    if not text:
        return {"error": "message 不可為空"}
    trip_id, photo_id = _like_ids(screen["folder"], screen["photo"])
    async with db.SessionLocal() as session:
        session.add(FrameMessage(
            frame_id=frame_id,
            user_label=screen["user"].strip()[:64],
            recipient=to,
            text=text,
            voice_id=screen["voice_id"],
            trip_id=trip_id,
            photo_id=photo_id,
        ))
        await session.commit()
    print(f"[message] frame {frame_id}: '{screen['user'] or '?'}' -> {to}: {text}")
    return {"ok": True, "note": f"已經把話留給{to}，他們的 App 會收到通知，"
                                "也聽得到你的聲音"}


def _voice_tool_slideshow(args: dict, state: dict) -> dict:
    cmd = _SLIDESHOW_CMDS.get(str(args.get("command") or "").strip())
    if cmd is None:
        return {"error": "command 要是 next / previous / pause / resume"}
    state["action"] = f"photo={cmd}"
    return {"ok": True, "note": _SLIDESHOW_NOTES[cmd]}


async def _voice_agent(frame_id: int, context: str, image_uri: str | None,
                       wav: bytes, turns: list[tuple[str, str]],
                       screen: dict, state: dict) -> tuple[str, str]:
    """Tool loop: the model hears the question, calls tools as it needs, and
    ends with respond(transcript, reply). Returns (transcript, reply).
    screen: what the frame is showing ({"folder", "photo", "user"}).
    Tools that need the board to do something put it in state["action"]
    (sent back as X-Frame-Action)."""
    messages = _voice_messages(VOICE_SYSTEM_TOOLS, context, image_uri, wav, turns)

    for _ in range(VOICE_AGENT_STEPS):
        m = await asyncio.to_thread(_voice_llm_step, messages)
        tool_calls = getattr(m, "tool_calls", None)

        if not tool_calls:                  # answered in plain text: take it
            reply = (m.content or "").strip()
            if reply:
                return "", reply
            raise RuntimeError("empty agent reply")

        # Keep the tool-call ids as they came: through the proxy they carry
        # Gemini's thought signature, which the next step must send back.
        messages.append({
            "role": "assistant", "content": m.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            if name == "respond":
                reply = str(args.get("reply") or "").strip()
                if reply:
                    return str(args.get("transcript") or "").strip(), reply
                result: dict = {"error": "reply 不可為空，請重新 respond"}
            elif name == "query_my_trips":
                result = await _tool_query_my_trips(args, frame_id=frame_id)
            elif name == "sync_frame":
                result = await _voice_tool_sync(frame_id, state)
            elif name == "like_photo":
                result = await _voice_tool_like(frame_id, screen)
            elif name == "show_album":
                result = await _voice_tool_show_album(frame_id, args, state)
            elif name == "control_slideshow":
                result = _voice_tool_slideshow(args, state)
            elif name == "read_story":
                result = await _voice_tool_read_story(screen, state)
            elif name == "leave_message":
                result = await _voice_tool_message(frame_id, args, screen)
            else:
                result = {"error": f"沒有 {name} 這個工具"}
            print(f"[voice] tool {name}({json.dumps(args, ensure_ascii=False)})")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    raise RuntimeError("voice agent exceeded max steps")


def _voice_llm_json(context: str, image_uri: str | None, wav: bytes,
                    turns: list[tuple[str, str]]) -> tuple[str, str]:
    """Fallback without tools — one call: hear the question, answer it.
    Returns (transcript, reply)."""
    resp = litellm.completion(
        model=LLM_MODEL,
        messages=_voice_messages(VOICE_SYSTEM_JSON, context, image_uri, wav, turns),
        response_format={"type": "json_object"},
        temperature=0.7,
        timeout=40,
        **_voice_llm_extra(),
    )
    # With audio in, the model now and then appends text after the JSON object
    # ("Extra data" from json.loads) - keep the first object, log the rest.
    text = resp.choices[0].message.content or ""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    data, end = json.JSONDecoder().raw_decode(text, start)
    if text[end:].strip():
        print(f"[voice] ignored text after the JSON: {text[end:].strip()[:120]!r}")
    return (str(data.get("transcript") or "").strip(),
            str(data.get("reply") or "").strip())


@app.post("/frames/{frame_id}/voice")
async def frame_voice(frame_id: int, request: Request, rate: int = 8000,
                      folder: str = "", photo: str = "", user: str = ""):
    """Body: raw 8-bit mu-law mono (application/octet-stream), sampled at
    `rate` Hz. folder/photo/user: same meaning as /like. Answers with the
    reply rendered as a story-rail TIM4 (see textimg.py) — even when the LLM
    fails, so the rail always shows 小憶 saying something."""
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty recording")
    if len(audio) > VOICE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="recording too long")
    if not 4000 <= rate <= 48000:
        raise HTTPException(status_code=400, detail=f"bad rate {rate}")

    async with db.SessionLocal() as session:
        await get_frame_or_404(session, frame_id)
        context, image_path = await _voice_context(session, folder, photo, user)

    voice_id = uuid.uuid4().hex[:12]
    wav = ulaw_to_wav(audio, rate)
    # Kept so a strange transcript can be listened back to.
    (AUDIO_DIR / f"voice_{voice_id}.wav").write_bytes(wav)
    image_uri = (await asyncio.to_thread(photo_data_uri, image_path, 512)
                 if image_path else None)

    t0 = time.time()
    turns = _voice_turns(frame_id)
    state: dict = {}                     # tools -> board: {"action": "..."}
    if os.environ.get("LLM_MOCK") == "1":
        question, reply = "（測試模式）", "我是小憶，現在是測試模式，還聽不懂你說的話喔。"
    else:
        try:
            screen = {"folder": folder, "photo": photo, "user": user,
                      "voice_id": voice_id}
            question, reply = await _voice_agent(
                frame_id, context, image_uri, wav, turns, screen, state)
        except Exception as e:  # noqa: BLE001 — retry without tools
            print(f"[voice] agent failed ({type(e).__name__}: {str(e)[:200]}); "
                  f"falling back to a plain answer")
            state.clear()   # a half-run tool loop must not act on the board
            try:
                question, reply = await asyncio.to_thread(
                    _voice_llm_json, context, image_uri, wav, turns)
            except Exception as e2:  # noqa: BLE001 — the rail still gets an answer
                print(f"[voice] LLM failed ({type(e2).__name__}: {str(e2)[:200]})")
                question, reply = "", "小憶這邊有點忙不過來，等一下再問我好嗎？"
    llm_ms = int((time.time() - t0) * 1000)

    reply = _NO_GLYPH_RE.sub("", reply).strip() or "我沒有聽清楚，可以再說一次嗎？"
    if question:
        _voice_remember(frame_id, question, reply)

    tim4 = render_reply_tim4(question, reply,
                             preview_png=AUDIO_DIR / f"voice_{voice_id}.png")
    # Kept so the board can re-fetch the image when the Wi-Fi module drops
    # bytes mid-download, instead of paying for the answer twice.
    (AUDIO_DIR / f"voice_{voice_id}.tim").write_bytes(tim4)

    # 小憶 says her answer too: synthesized after this response goes out, and
    # fetched by the board from .../voice/{id}/audio while it draws the rail.
    VOICE_REPLY_AUDIO[voice_id] = {"status": "pending", "path": None, "error": ""}
    asyncio.get_running_loop().create_task(_voice_reply_tts(voice_id, reply))
    print(f"[voice] frame {frame_id}: {len(audio)} B ({len(audio) * 1000 // rate} ms) "
          f"at {folder or '-'}/{photo or '-'} user='{user or '?'}' "
          f"photo={'yes' if image_uri else 'no'} llm={llm_ms}ms id={voice_id}")
    print(f"[voice] heard: {question or '(nothing)'}")
    print(f"[voice] reply: {reply}")
    headers = {"X-Voice-Id": voice_id}
    if state.get("action"):
        # ASCII token the board acts on after showing the answer (VoiceChat.c)
        headers["X-Frame-Action"] = state["action"]
        print(f"[voice] frame action: {state['action']}")
    return Response(content=tim4, media_type="application/octet-stream",
                    headers=headers)


# voice_id -> {"status": pending|ready|error, "path": Path, "error": str}
VOICE_REPLY_AUDIO: dict[str, dict] = {}
VOICE_AUDIO_WAIT_S = 20          # the board asks right after the answer lands


async def _voice_reply_tts(voice_id: str, text: str) -> None:
    """小憶's answer as 8 kHz mu-law for the frame to play, in her App voice.
    Runs after the answer has been sent, so it costs the reply no latency."""
    import edge_tts

    mp3 = AUDIO_DIR / f"reply_{voice_id}.mp3"
    ulaw = AUDIO_DIR / f"reply_{voice_id}.ulaw"
    job = VOICE_REPLY_AUDIO[voice_id]
    try:
        await edge_tts.Communicate(
            text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH).save(str(mp3))
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(mp3),
            "-ar", "8000", "-ac", "1", "-f", "mulaw", str(ulaw),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace")[-200:])
        job.update(status="ready", path=ulaw)
        print(f"[voice] {voice_id} speech ready: {ulaw.stat().st_size} bytes")
    except Exception as e:  # noqa: BLE001 — the rail still has the text
        job.update(status="error", error=str(e))
        print(f"[voice] {voice_id} TTS failed: {e}")
    finally:
        mp3.unlink(missing_ok=True)


@app.get("/frames/{frame_id}/voice/{voice_id}/audio")
async def frame_voice_audio(frame_id: int, voice_id: str):
    """小憶 saying her answer (8 kHz mu-law, as Audio.c plays). Held until the
    TTS that started with the answer is done."""
    job = VOICE_REPLY_AUDIO.get(voice_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown voice_id")

    deadline = time.time() + VOICE_AUDIO_WAIT_S
    while job["status"] == "pending" and time.time() < deadline:
        await asyncio.sleep(0.1)
    if job["status"] != "ready":
        raise HTTPException(status_code=503,
                            detail=f"speech not ready: {job.get('error', 'timeout')}")
    return FileResponse(job["path"], media_type="application/octet-stream")


@app.get("/frames/{frame_id}/voice/{voice_id}/image")
def frame_voice_image(frame_id: int, voice_id: str):
    """The answer as the rail image again — the board re-fetches it when the
    Wi-Fi module drops bytes, instead of asking the LLM all over again."""
    if not re.fullmatch(r"[0-9a-f]{12}", voice_id):
        raise HTTPException(status_code=404, detail="unknown voice_id")
    tim = AUDIO_DIR / f"voice_{voice_id}.tim"
    if not tim.exists():
        raise HTTPException(status_code=404, detail="unknown voice_id")
    return FileResponse(tim, media_type="application/octet-stream")


@app.get("/frames/{frame_id}/voice/{voice_id}/preview")
def frame_voice_preview(frame_id: int, voice_id: str):
    """PNG of a reply as the rail shows it (browser check only)."""
    if not re.fullmatch(r"[0-9a-f]{12}", voice_id):
        raise HTTPException(status_code=404, detail="unknown voice_id")
    png = AUDIO_DIR / f"voice_{voice_id}.png"
    if not png.exists():
        raise HTTPException(status_code=404, detail="unknown voice_id")
    return FileResponse(png, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
