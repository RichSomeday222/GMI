# src/query_engine.py

import os, json, requests, faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# —— 加载环境 & 常量 —— #
load_dotenv()
API_KEY = os.getenv("GMI_API_KEY")
if not API_KEY:
    raise RuntimeError("请在 .env 中设置 GMI_API_KEY")

API_URL = "https://api.gmi-serving.com/v1/chat/completions"
ROOT = Path(__file__).parent.parent

# —— 准备索引 & 文本 & 嵌入模型 —— #
index = faiss.read_index(str(ROOT / "index" / "resume.index"))
with open(ROOT / "embeddings" / "texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

# —— 检索函数 —— #
def retrieve(query: str, top_k: int = 3) -> list[str]:
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    return [texts[i] for i in I[0]]

# —— Prompt 构造 —— #
def build_prompt(query: str, retrieved_texts: list[str]) -> str:
    prompt = "以下是与你问题相关的简历片段：\n"
    for idx, chunk in enumerate(retrieved_texts, 1):
        prompt += f"{idx}. {chunk}\n"
    prompt += f"\n请基于上述信息，回答问题：“{query}”"
    return prompt

# —— 调用 GMI REST 接口 —— #
def answer_query(query: str, top_k: int = 3) -> str:
    hits   = retrieve(query, top_k)
    prompt = build_prompt(query, hits)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# —— 本地测试 —— #
if __name__ == "__main__":
    q = "我的项目经验都有哪些亮点？"
    print(">>", answer_query(q))
