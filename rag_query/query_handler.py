from chromadb.utils import embedding_functions
import chromadb
from utils.llama_cpp_runner import run_llama_ollama  # llama.cpp CLI wrapper

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_or_create_collection(
    name="video_chunks",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
)

def retrieve_chunks(query: str, top_k: int = 5):
    return collection.query(query_texts=[query], n_results=top_k)

def generate_response(query: str, contexts: list[str], timestamps: list[str]) -> str:
    prompt = f"""
You are an AI assistant that understands video content.

Below are the extracted multimodal segments (audio + visual) from a video:
{chr(10).join([f"[{ts}] {ctx}" for ts, ctx in zip(timestamps, contexts)])}

Now, answer this question clearly using the most relevant information above:
"{query}"

Respond concisely, and mention timestamps when applicable.
"""
    return run_llama_ollama(prompt)
