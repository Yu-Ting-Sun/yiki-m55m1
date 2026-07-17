# SQLite + SQLAlchemy(async) layer for the 憶起 app backend.
#
# Tables (spec'd by the app plan):
#   trips:      id, title, start_time, end_time, distance_m, story_text, created_at
#   gps_points: id, trip_id, lat, lng, timestamp
#   photos:     id, trip_id, filename, lat, lng, timestamp
#   frames:     id, pair_code, name, last_sync
#
# All datetimes are stored naive-UTC; the API layer serializes them as ISO-8601
# strings with a trailing "Z" so Flutter's DateTime.parse() gets the zone right.
import math
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DB_PATH = Path(__file__).parent / "yiki.db"

engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    start_time: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    distance_m: Mapped[float] = mapped_column(Float, default=0.0)
    story_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: utcnow())

    points: Mapped[list["GpsPoint"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", order_by="GpsPoint.timestamp"
    )
    photos: Mapped[list["Photo"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", order_by="Photo.timestamp"
    )


class GpsPoint(Base):
    __tablename__ = "gps_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime)

    trip: Mapped[Trip] = relationship(back_populates="points")


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    lat: Mapped[float | None] = mapped_column(Float, default=None)
    lng: Mapped[float | None] = mapped_column(Float, default=None)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    trip: Mapped[Trip] = relationship(back_populates="photos")


class TripSpot(Base):
    """使用者在記錄中從「附近景點」親手收藏進旅程的地點。"""

    __tablename__ = "trip_spots"

    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String(50), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: utcnow())


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    pair_code: Mapped[str] = mapped_column(String(6), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100), default="智慧相框")
    last_sync: Mapped[datetime | None] = mapped_column(DateTime, default=None)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def utcnow() -> datetime:
    """Naive UTC now — matches how all DateTime columns are stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_naive_utc(dt: datetime) -> datetime:
    """Normalize an incoming (possibly zone-aware) datetime to naive UTC."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def iso_z(dt: datetime | None) -> str | None:
    """Naive-UTC datetime -> '2026-07-08T12:34:56.789Z' (None passes through)."""
    return None if dt is None else dt.isoformat(timespec="milliseconds") + "Z"


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def path_distance_m(points: list[tuple[float, float]]) -> float:
    """Sum of segment distances over an ordered (lat, lng) list."""
    return sum(
        haversine_m(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )
