import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: list[dict]) -> tuple:
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks


def retrieve(query: str, index, chunks: list[dict], top_k: int = 4) -> list[dict]:
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context_chunks])
    prompt = f"""You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always mention which page number your answer comes from.

Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
