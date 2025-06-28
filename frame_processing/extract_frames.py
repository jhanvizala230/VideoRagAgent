import cv2
import os
from pathlib import Path
from typing import List, Tuple

def extract_frames(video_path: str, output_dir: str, interval_sec: int = 2) -> List[Tuple[str, float]]:
    """
    Extract frames every `interval_sec` seconds from the video.
    
    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to store extracted frames.
        interval_sec (int): Interval in seconds to extract frames.
    
    Returns:
        List of tuples: (frame_path, timestamp_sec)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_interval = int(fps * interval_sec)

    frame_idx = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            frame_filename = f"frame_{int(timestamp)}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames.append((frame_path, round(timestamp, 2)))

        frame_idx += 1

    cap.release()
    return saved_frames
