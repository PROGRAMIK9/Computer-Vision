"""Browser-compatible MP4 conversion and byte-range response helpers."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse


def codec_name(path: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def browser_mp4(source: Path) -> Path:
    """Return an H.264/yuv420p MP4 path, transcoding the OpenCV mp4v output if needed."""
    codec = codec_name(source)
    if codec == "h264":
        return source
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=503,
            detail="This event uses an MP4 codec the browser may not play. Install ffmpeg on the backend host to convert it to H.264.",
        )
    output = source.with_name("video.browser.mp4")
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(source), "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not output.is_file():
        output.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video conversion to browser-compatible H.264 failed: {result.stderr[-500:]}")
    return output


def range_file_response(path: Path, request: Request) -> StreamingResponse:
    size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Content-Disposition": 'inline; filename="event.mp4"'}
    if not range_header:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_chunks(path, 0, size), media_type="video/mp4", headers=headers)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(status_code=416, detail="Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
    if not match.group(1):
        suffix_size = int(match.group(2))
        start, end = max(0, size - suffix_size), size - 1
    else:
        start = int(match.group(1))
        end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
    if start >= size or end < start:
        raise HTTPException(status_code=416, detail="Range not satisfiable", headers={"Content-Range": f"bytes */{size}"})
    length = end - start + 1
    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
    return StreamingResponse(_chunks(path, start, length), status_code=206, media_type="video/mp4", headers=headers)


def _chunks(path: Path, start: int, length: int, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as video:
        video.seek(start)
        remaining = length
        while remaining:
            chunk = video.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

