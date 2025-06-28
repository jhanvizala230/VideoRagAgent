from faster_whisper import WhisperModel

def transcribe_audio(audio_path: str, model_size: str = "base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
    
    results = []
    for segment in segments:
        results.append({
            "text": segment.text.strip(),
            "start": round(segment.start, 2),
            "end": round(segment.end, 2)
        })
    return results
