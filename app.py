import streamlit as st
import os
import tempfile
import json
import sys
import hashlib

sys.path.append(os.path.abspath("."))

# Internal pipeline modules
from audio_processing.extract_audio import extract_audio
from audio_processing.transcribe_whisper import transcribe_audio
from frame_processing.extract_frames import extract_frames
from frame_processing.caption_frames import caption_all_frames
from chunk_builder.merge_chunks import merge_transcript_and_captions
from embedding.embed_and_store import embed_and_store
from rag_query.query_handler import retrieve_chunks, generate_response

# Streamlit setup
st.set_page_config(page_title="Video RAG Pipeline", layout="wide")
st.title("🎥 Video Understanding & Timestamped RAG")

# Folder setup
RAW_DIR = "data/raw/"
FRAMES_DIR = "data/frames/"
TRANSCRIPTS_DIR = "data/transcripts/"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# Utility: hash video to identify and cache
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Upload
uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
        print(video_path)

    st.video(video_path)

    # Hash for identification
    video_hash = get_file_hash(video_path)
    TRANSCRIPT_JSON = f"{TRANSCRIPTS_DIR}/transcript_{video_hash}.json"
    CAPTIONS_JSON = f"{TRANSCRIPTS_DIR}/frame_captions_{video_hash}.json"
    MERGED_JSON = f"{TRANSCRIPTS_DIR}/merged_chunks_{video_hash}.json"
    CACHE_FILE = f"{TRANSCRIPTS_DIR}/cache_{video_hash}.json"
    FRAME_HASH_FOLDER = os.path.join(FRAMES_DIR, video_hash)
    AUDIO_PATH = os.path.join(RAW_DIR, f"audio_{video_hash}.wav")

    os.makedirs(FRAME_HASH_FOLDER, exist_ok=True)

    # If already processed
    if os.path.exists(CACHE_FILE) and os.path.exists(MERGED_JSON):
        st.success("✅ This video was already processed. Loading cached results...")
        with open(MERGED_JSON, "r") as f:
            merged_chunks = json.load(f)
        st.json(merged_chunks[:3])

    elif st.button("🚀 Run Full Pipeline"):
        with st.spinner("🔊 Extracting audio..."):
            audio_path = extract_audio(video_path, AUDIO_PATH)

        with st.spinner("📝 Transcribing audio..."):
            transcript = transcribe_audio(audio_path)
            with open(TRANSCRIPT_JSON, "w") as f:
                json.dump(transcript, f, indent=2)
            st.success("Transcript saved")
            st.json(transcript[:3])

        with st.spinner("🖼️ Extracting frames..."):
            frames = extract_frames(video_path, FRAME_HASH_FOLDER, interval_sec=2)
            st.success(f"{len(frames)} frames extracted")

        with st.spinner("🧠 Captioning frames using BLIP-2..."):
            captions = caption_all_frames(FRAME_HASH_FOLDER)
            with open(CAPTIONS_JSON, "w") as f:
                json.dump(captions, f, indent=2)
            st.success("Captions saved")
            st.json(dict(list(captions.items())[:3]))

        with st.spinner("🔗 Merging audio and visual data..."):
            merged_chunks = merge_transcript_and_captions(
                transcript_path=TRANSCRIPT_JSON,
                captions_path=CAPTIONS_JSON,
                output_path=MERGED_JSON
            )
            st.success("Multimodal merged chunks created")
            st.json(merged_chunks[:3])

        with st.spinner("📦 Embedding & storing in ChromaDB..."):
            embed_and_store(MERGED_JSON, persist_dir="vector_store")
            st.success("Embeddings stored in vector DB")

        # Save status
        with open(CACHE_FILE, "w") as f:
            json.dump({"status": "processed"}, f)

        st.success("✅ Pipeline complete and cached!")

    # Query interface
st.markdown("### 💬 Ask a question about the video:")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("🔍 Retrieving relevant segments..."):
        results = retrieve_chunks(query)
        docs = results["documents"][0]
        times = [f"{m['start']}s - {m['end']}s" for m in results["metadatas"][0]]

    with st.spinner("🧠 Generating response with LLaMA 3.2 (CLI)..."):
        answer = generate_response(query, docs, times)
        st.markdown("**Answer:**")
        st.markdown(answer)

# Footer
st.markdown("---")
st.markdown("Built with Whisper, BLIP-2, ChromaDB, and LLaMA 3.2 (CLI via llama.cpp)")
