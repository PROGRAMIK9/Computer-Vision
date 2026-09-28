"""FastAPI backend with SQLite persistence for cameras, events, and event videos."""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from edge.config import settings

app = FastAPI(title="Edge AI Camera API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets", check_dir=False), name="frontend-assets")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page():
    index = _frontend_dist / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse("<h2>React dashboard not built yet</h2><p>Start it with <code>npm run dev</code> in frontend/.</p>")


class CameraCreate(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    name: str = Field(min_length=1, max_length=160)
    device_id: str | None = Field(default=None, max_length=160)


def _connect() -> sqlite3.Connection:
    settings.backend_db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.backend_db_path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def _init_db() -> None:
    with _connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, device_id TEXT,
            status TEXT NOT NULL DEFAULT 'offline', created_at TEXT NOT NULL,
            last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL, duration REAL NOT NULL, video_path TEXT,
            metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(camera_id) REFERENCES cameras(id)
        );
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
            class_name TEXT NOT NULL, confidence REAL NOT NULL,
            timestamp TEXT NOT NULL, bbox_json TEXT,
            FOREIGN KEY(event_id) REFERENCES events(id)
        );
        CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at DESC);
        """)


@app.on_event("startup")
def startup() -> None:
    _init_db()


def _camera_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    if result.get("last_seen"):
        try:
            seen = datetime.fromisoformat(result["last_seen"])
            age = (datetime.now(timezone.utc) - seen).total_seconds()
            result["status"] = "online" if age <= 45 else "offline"
        except (TypeError, ValueError):
            result["status"] = "offline"
    return result


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = json.loads(row["metadata_json"])
    result["video_available"] = bool(row["video_path"] and Path(row["video_path"]).is_file())
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "edge-ai-camera-api"}


@app.post("/api/cameras", status_code=201)
def register_camera(camera: CameraCreate) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        try:
            db.execute("INSERT INTO cameras(id,name,device_id,status,created_at,last_seen) VALUES(?,?,?,'offline',?,NULL)",
                       (camera.id, camera.name.strip(), camera.device_id or camera.id, now))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Camera ID is already registered")
        row = db.execute("SELECT * FROM cameras WHERE id=?", (camera.id,)).fetchone()
        return _camera_dict(row)


@app.get("/api/cameras")
def list_cameras() -> list[dict[str, Any]]:
    with _connect() as db:
        return [_camera_dict(row) for row in db.execute("SELECT * FROM cameras ORDER BY created_at DESC")]


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str) -> dict[str, Any]:
    with _connect() as db:
        row = db.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        return _camera_dict(row)


@app.post("/api/cameras/{camera_id}/heartbeat")
def camera_heartbeat(camera_id: str) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        updated = db.execute("UPDATE cameras SET status='online',last_seen=? WHERE id=?", (now, camera_id)).rowcount
        if not updated:
            raise HTTPException(status_code=404, detail="Camera not registered")
    return {"status": "online", "last_seen": now}


@app.get("/api/events")
def list_events(camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 500))
    with _connect() as db:
        if camera_id:
            rows = db.execute("SELECT * FROM events WHERE camera_id=? ORDER BY started_at DESC LIMIT ?",
                              (camera_id, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM events ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [_event_dict(row) for row in rows]


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> dict[str, Any]:
    with _connect() as db:
        row = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        result = _event_dict(row)
        result["detections"] = [dict(d) | ({"bbox": json.loads(d["bbox_json"])} if d["bbox_json"] else {})
                                for d in db.execute("SELECT * FROM detections WHERE event_id=? ORDER BY timestamp", (event_id,))]
        return result


@app.post("/api/events")
async def create_event(request: Request) -> dict[str, Any]:
    try:
        event = await request.json()
        required = ("event_id", "camera_id", "started_at", "ended_at", "duration")
        if any(key not in event for key in required):
            raise HTTPException(status_code=422, detail=f"Required fields: {', '.join(required)}")
        event_id = str(event["event_id"])
        camera_id = str(event["camera_id"])
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(event)
        with _connect() as db:
            camera = db.execute("SELECT id FROM cameras WHERE id=?", (camera_id,)).fetchone()
            if camera is None:
                # Edge may predate website registration; create a visible camera entry.
                db.execute("INSERT INTO cameras(id,name,device_id,status,created_at,last_seen) VALUES(?,?,?,'online',?,?)",
                           (camera_id, event.get("camera_name", camera_id), camera_id, now, now))
            else:
                db.execute("UPDATE cameras SET status='online',last_seen=? WHERE id=?", (now, camera_id))
            db.execute("""INSERT INTO events(id,camera_id,started_at,ended_at,duration,video_path,metadata_json,created_at)
                VALUES(?,?,?,?,?,NULL,?,?) ON CONFLICT(id) DO UPDATE SET metadata_json=excluded.metadata_json""",
                (event_id, camera_id, event["started_at"], event["ended_at"], float(event["duration"]), metadata_json, now))
            for detection in event.get("detections", []):
                db.execute("""INSERT INTO detections(event_id,class_name,confidence,timestamp,bbox_json)
                    SELECT ?,?,?,?,? WHERE NOT EXISTS(SELECT 1 FROM detections WHERE event_id=? AND timestamp=? AND class_name=? AND confidence=?)""",
                    (event_id, detection.get("class_name", "unknown"), float(detection.get("confidence", 0)),
                     detection.get("timestamp", event["started_at"]), json.dumps(detection["bbox"]) if detection.get("bbox") else None,
                     event_id, detection.get("timestamp", event["started_at"]), detection.get("class_name", "unknown"),
                     float(detection.get("confidence", 0))))
        return {"status": "accepted", "event_id": event_id}
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid event payload: {exc}") from exc


@app.put("/api/events/{event_id}/video")
async def upload_event_video(event_id: str, request: Request) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", event_id):
        raise HTTPException(status_code=400, detail="Invalid event ID")
    with _connect() as db:
        if db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Create event metadata before uploading video")
    media_dir = settings.backend_media_dir / f"event_{event_id}"
    media_dir.mkdir(parents=True, exist_ok=True)
    video_path = media_dir / "video.mp4"
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty video upload")
    video_path.write_bytes(data)
    with _connect() as db:
        db.execute("UPDATE events SET video_path=? WHERE id=?", (str(video_path), event_id))
    return {"status": "stored", "event_id": event_id}


@app.get("/api/events/{event_id}/video")
def get_event_video(event_id: str):
    with _connect() as db:
        row = db.execute("SELECT video_path FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        if not row["video_path"] or not Path(row["video_path"]).is_file():
            raise HTTPException(status_code=404, detail="Event video not uploaded")
        return FileResponse(row["video_path"], media_type="video/mp4", filename="video.mp4")
