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
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import litellm  # heavy import (~10 s); pay it at startup, not on first /generate
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

import db
from db import GpsPoint, Photo, Trip
from textimg import render_story_tim4


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_db()
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

async def tts_job(audio_id: str, text: str):
    """edge-tts -> mp3, then ffmpeg -> WAV PCM 16 kHz / 16-bit / mono.

    -bitexact keeps the WAV header at the plain 44-byte RIFF layout (no LIST
    metadata chunk) so a naive header parser on the MCU still finds "data".
    """
    job = AUDIO_JOBS[audio_id]
    mp3_path = AUDIO_DIR / f"{audio_id}.mp3"
    wav_path = AUDIO_DIR / f"{audio_id}.wav"
    try:
        import edge_tts

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

        job["status"] = "ready"
        print(f"[tts] {audio_id} ready: {wav_path.stat().st_size} bytes")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[tts] {audio_id} FAILED: {e}")
    finally:
        mp3_path.unlink(missing_ok=True)


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
        await session.delete(trip)  # ORM cascade removes points + photos rows
        await session.commit()
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


# ================================================================ App: story
# 用旅程「真實資料」生成遊記（改造 Day 3 /generate：假資料 → 查 DB）。
# 沒有地名資料（Stage 4 才有景點），prompt 明確要求不得編造地名。

WEEKDAY_ZH = "一二三四五六日"


def llm_json(prompt: str) -> dict:
    """One-shot LLM call returning parsed JSON. Same routing/env as call_llm."""
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
    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            timeout=30,
            **extra,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


def build_trip_story_prompt(trip: Trip) -> str:
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
    if trip.photos:
        parts.append(f"沿途拍了:{len(trip.photos)} 張照片")

    return (
        "你是家庭旅遊回憶 App「憶起」的遊記寫手。"
        "根據以下真實行程資料，寫一段給家人回味的遊記。\n"
        + "\n".join(parts)
        + "\n\n要求：\n"
        "1. story：繁體中文 100 到 150 字，第一人稱「我們」，"
        "語氣溫暖親切，像寫在家庭相簿裡的回憶。\n"
        "2. 只根據上面給的資料寫；「沒有」提供地名，所以不要編造任何地名、"
        "店名或具體事件，可以描寫時間帶的氛圍、步行的心情與陪伴的感覺。\n"
        "3. 不要標題、不要引號，直接是遊記內容。\n"
        '只輸出 JSON：{"story": "..."}'
    )


class StoryUpdate(BaseModel):
    story_text: str


@app.post("/trips/{trip_id}/story/generate")
async def generate_trip_story(trip_id: int):
    async with db.SessionLocal() as session:
        trip = await get_trip_or_404(session, trip_id, with_children=True)

        if os.environ.get("LLM_MOCK") == "1":
            story = (
                f"這天我們一起出門走了走，{trip.title or '這趟小旅程'}雖然不長，"
                "沿路的風、路邊的花草，還有彼此的笑聲，都讓平凡的一天"
                "變得暖暖的。回家的路上大家都說，下次還要再一起出來走走。"
            )
        else:
            t0 = time.time()
            data = await asyncio.to_thread(llm_json, build_trip_story_prompt(trip))
            story = str(data.get("story", "")).strip()
            print(f"[story] trip {trip_id} llm={int((time.time() - t0) * 1000)}ms")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
