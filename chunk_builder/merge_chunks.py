import json
import os
from typing import List, Dict

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def find_nearest_caption(timestamp: float, caption_dict: Dict[float, str]) -> str:
    # Find the caption whose timestamp is closest to the transcript start time
    closest_ts = min(caption_dict.keys(), key=lambda t: abs(t - timestamp))
    return caption_dict[closest_ts]

def merge_transcript_and_captions(
    transcript_path: str,
    captions_path: str,
    output_path: str
) -> List[Dict]:
    transcript = load_json(transcript_path)
    captions = load_json(captions_path)
    captions = {float(k): v for k, v in captions.items()}

    merged = []
    for chunk in transcript:
        start = chunk["start"]
        end = chunk["end"]
        text = chunk["text"]
        visual_caption = find_nearest_caption(start, captions)

        merged.append({
            "text": text,
            "image_caption": visual_caption,
            "start": round(start, 2),
            "end": round(end, 2)
        })

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged
