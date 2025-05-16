# src/query_engine.py

import os, json, requests, faiss
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")



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
bm25_corpus = [word_tokenize(text) for text in source_texts]
bm25 = BM25Okapi(bm25_corpus)
# —— System message for GMI prompts —— #

def system_message() -> Dict[str,str]:
    return {
        "role": "system",
        "content": "You are a helpful assistant that answers based only on provided context."
    }

async def generate_hyde_doc_async(query: str) -> str:
    """
    调用 GMI 生成一个 '假想答案'Hypothetical Document Expansion。
    """
    hyde_prompt = f"""
Please generate a short hypothetical answer to the following question based on 
your common sense or general knowledge.
Do not explain the reason, just output the answer itself:
Question: {query}
"""
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            system_message(),
            {"role": "user", "content": hyde_prompt}
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()

# —— Vector search —— #
def retrieve(query: str, top_k: int = 3) -> List[str]:
    # 1. info_chunk 保底
    candidates = [ source_texts[0] ]  # 先放个人信息块

    # 2. 原始 dense+BM25 检索
    vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    _, vec_idxs  = index.search(vec, top_k)
    bm25_scores = bm25.get_scores(word_tokenize(query))
    bm25_idxs   = np.argsort(bm25_scores)[::-1][:top_k]

    hits = list(dict.fromkeys(list(vec_idxs[0]) + list(bm25_idxs)))
    # 3. 把 hits 中除了 0 号以外的依次加进来
    for idx in hits:
        if idx != 0 and len(candidates) < top_k:
            candidates.append(source_texts[idx])
    return candidates


# —— Prompt construction —— #
def build_prompt(query: str, chunks: List[str], hyde: str = None) -> str:
    prompt = "Here is a snippet of my resume related to your question: \n"
    for i, c in enumerate(chunks, 1):
        prompt += f"{i}. {c}\n"
    if hyde:
        prompt += f"\nHypothetical Document：{hyde}\n"
    prompt += f"\nPlease answer the questions based on the above information:“{query}”"
    return prompt

# —— Async Call GMI REST API —— #
async def answer_query_async(query: str, history: List[Dict[str, str]] = None, top_k: int = 3) -> str:
    """异步版本的 RAG 查询（含 HyDE）"""
    history = history or []
    # —— HyDE 生成 —— #
    hyde_doc = await generate_hyde_doc_async(query)

    # —— 检索 —— #
    real_hits = retrieve(query, top_k)
    hyde_hits = retrieve(hyde_doc, top_k)
    all_hits  = list(dict.fromkeys(real_hits + hyde_hits))[:top_k]
    # —— 构造 messages，把 history 先放进去 —— #
    messages = [ system_message() ]
    messages += history 
    # 把检索上下文包装成一条新的 user 消息
    prompt_content = build_prompt(query, all_hits, hyde=hyde_doc)
    messages.append({"role":"user", "content": prompt_content})


    # 5. 调用 GMI
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":    "deepseek-ai/DeepSeek-R1",
        "messages": messages
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
        
# —— Original synchronous function (now using async under the hood) —— #
def answer_query(query: str, top_k: int = 3) -> str:
    """同步版本的查询回答函数（内部使用异步函数）"""
    return asyncio.run(answer_query_async(query, top_k))

# —— Batch processing multiple queries concurrently —— #
async def batch_answer_queries(queries: List[str], top_k: int = 3) -> List[str]:
    """并发处理多个查询"""
    tasks = [answer_query_async(query, top_k=top_k) for query in queries]
    return await asyncio.gather(*tasks)

# —— Performance monitoring wrapper —— #
async def timed_answer_query(query: str, top_k: int = 3) -> Dict[str, Any]:
    """带性能监控的异步查询函数"""
    start_time = asyncio.get_event_loop().time()
    
    # 检索相关文本
    retrieval_start = asyncio.get_event_loop().time()
    related_texts = retrieve(query, top_k)
    retrieval_time = asyncio.get_event_loop().time() - retrieval_start
    
    # 构建提示
    prompt_start = asyncio.get_event_loop().time()
    prompt = build_prompt(query, related_texts)
    prompt_time = asyncio.get_event_loop().time() - prompt_start
    
    # 调用API
    api_start = asyncio.get_event_loop().time()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            result = await response.json()
            answer = result["choices"][0]["message"]["content"]
    
    api_time = asyncio.get_event_loop().time() - api_start
    total_time = asyncio.get_event_loop().time() - start_time
    
    return {
        "answer": answer,
        "metrics": {
            "retrieval_time": retrieval_time,
            "prompt_time": prompt_time,
            "api_time": api_time,
            "total_time": total_time
        }
    }
    
    
    


# —— Command line interface —— #
async def run_cli():
    """异步命令行界面，支持多轮上下文"""
    history: List[Dict[str, str]] = []
    print("Multi-round resume question-answering system (enter 'quit' to exit)")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            print("Bye！")
            break

        print("Querying…")
        try:
            # 1 带上 history 调用核心接口
            answer = await answer_query_async(query, history, top_k=3)

            # 2 打印回答
            print("\nAssistant:", answer)

            # 3 将本轮问答追加到 history
            history.append({"role": "user",      "content": query})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"Error: {e}")
            
# —— Local test —— #
if __name__ == "__main__":
    if asyncio.run(asyncio.sleep(0)) is None:  # 检查是否运行在支持异步的环境
        print("使用异步模式运行...")
        asyncio.run(run_cli())
    else:
        print("环境不支持异步，使用同步模式...")
        sample_query = "What are the highlights of my project experience?"
        print(">>", answer_query(sample_query))