# frame_processing/caption_frames.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

def load_blip2_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

def caption_frame(image_path: str, processor, model, device) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()

def caption_all_frames(frames_dir: str):
    processor, model, device = load_blip2_model()
    captions = {}

    for filename in sorted(os.listdir(frames_dir)):
        if not filename.endswith(".jpg"):
            continue
        timestamp = float(filename.replace("frame_", "").replace("s.jpg", ""))
        image_path = os.path.join(frames_dir, filename)
        caption = caption_frame(image_path, processor, model, device)
        captions[timestamp] = caption

    return captions
