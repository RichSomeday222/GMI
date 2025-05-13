# src/query_engine.py

import os, json, requests, faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# —— Load environment & constants —— #
load_dotenv()
API_KEY = os.getenv("GMI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GMI_API_KEY in your .env file")

API_URL = "https://api.gmi-serving.com/v1/chat/completions"
ROOT = Path(__file__).parent.parent

# —— Load FAISS index, source texts, and embedding model —— #
index = faiss.read_index(str(ROOT / "index" / "resume.index"))
with open(ROOT / "embeddings" / "texts.json", "r", encoding="utf-8") as f:
    source_texts = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

# —— Vector search —— #
def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    return [source_texts[i] for i in indices[0]]

# —— Prompt construction —— #
def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    prompt = "Here are some resume snippets related to your question:\n"
    for idx, chunk in enumerate(retrieved_chunks, 1):
        prompt += f"{idx}. {chunk}\n"
    prompt += f"\nBased on the information above, please answer the question: \"{query}\""
    return prompt

# —— Call GMI REST API —— #
def answer_query(query: str, top_k: int = 3) -> str:
    related_texts = retrieve(query, top_k)
    prompt = build_prompt(query, related_texts)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# —— Local test —— #
if __name__ == "__main__":
    sample_query = "What are the highlights of my project experience?"
    print(">>", answer_query(sample_query))
