import os
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List

from .graph_build import get_graph
from .video_utils import extract_frames

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="ReVive AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    work_dir = tempfile.mkdtemp(prefix="revive_")
    image_paths = []

    try:
        for upload in files:
            ext = Path(upload.filename or "").suffix.lower()
            dest_path = os.path.join(work_dir, f"{uuid.uuid4().hex}{ext}")

            with open(dest_path, "wb") as out_file:
                shutil.copyfileobj(upload.file, out_file)

            if ext in IMAGE_EXTS:
                image_paths.append(dest_path)
            elif ext in VIDEO_EXTS:
                frames = extract_frames(dest_path, work_dir, max_frames=5)
                image_paths.extend(frames)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {upload.filename}",
                )

        if not image_paths:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any usable images from the upload.",
            )

        graph = get_graph()
        result = graph.invoke({"image_paths": image_paths})

        return JSONResponse({
            "detected_items": result.get("detected_items", []),
            "classification": result.get("classification", {}),
            "recommendations": result.get("recommendations", {}),
            "pickup_plan": result.get("pickup_plan", {}),
            "sustainability_metrics": result.get("sustainability_metrics", {}),
            "report_text": result.get("report", ""),
            "images_analyzed": len(image_paths),
        })

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# Serve the frontend (index.html, css, js) from the same server so the whole
# app can be run with a single `uvicorn` command.
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
