import ffmpeg

def extract_audio(video_path: str, output_audio_path: str = "output.wav"):
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(overwrite_output=True)
    )
    return output_audio_path
