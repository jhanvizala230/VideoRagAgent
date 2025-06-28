import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

def load_merged_chunks(path: str) -> List[Dict]:
    """Loads the merged multimodal chunks from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def create_chroma_collection(persist_dir="vector_store"):
    """
    Creates or retrieves a persistent ChromaDB collection for storing video chunks.
    Uses SentenceTransformer for embedding.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="video_chunks",
        embedding_function=embedding_function
    )
    return collection

def embed_and_store(merged_chunks_path: str, persist_dir: str = "vector_store"):
    print("🔍 Loading merged chunks...")
    chunks = load_merged_chunks(merged_chunks_path)

    print("🔠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        combined_text = f"{chunk['text']} [VISUAL]: {chunk['image_caption']}"
        documents.append(combined_text)
        metadatas.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "image_caption": chunk["image_caption"]
        })
        ids.append(f"chunk-{i}")

    print("📦 Generating embeddings...")
    embeddings = model.encode(documents, convert_to_numpy=True)

    print("📥 Storing embeddings in ChromaDB...")
    collection = create_chroma_collection(persist_dir)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Successfully stored {len(documents)} chunks in ChromaDB at '{persist_dir}'")
    
