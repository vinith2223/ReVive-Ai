"""Utility to sample a handful of frames from an uploaded video."""

import os
import uuid


def extract_frames(video_path: str, out_dir: str, max_frames: int = 5) -> list:
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames <= 0:
        cap.release()
        return []

    step = max(total_frames // max_frames, 1)
    frame_paths = []

    idx = 0
    saved = 0
    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        frame_path = os.path.join(out_dir, f"frame_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        idx += step
        saved += 1
        if idx >= total_frames:
            break

    cap.release()
    return frame_paths
