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

# —— Vector search —— #
def retrieve(query: str, top_k: int = 3) -> list[str]:
    # 向量检索
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    _, vec_indices = index.search(query_vec, top_k)

    # BM25 检索
    query_tokens = word_tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # 合并 index，去重
    combined = list(dict.fromkeys(list(vec_indices[0]) + list(bm25_indices)))[:top_k]

    return [source_texts[i] for i in combined]


# —— Prompt construction —— #
def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    prompt = "Here are some resume snippets related to your question:\n"
    for idx, chunk in enumerate(retrieved_chunks, 1):
        prompt += f"{idx}. {chunk}\n"
    prompt += f"\nBased on the information above, please answer the question: \"{query}\""
    return prompt

# —— Async Call GMI REST API —— #
async def answer_query_async(query: str, top_k: int = 3) -> str:
    """异步版本的查询回答函数"""
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

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API调用失败，状态码: {response.status}, 错误: {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]

# —— Original synchronous function (now using async under the hood) —— #
def answer_query(query: str, top_k: int = 3) -> str:
    """同步版本的查询回答函数（内部使用异步函数）"""
    return asyncio.run(answer_query_async(query, top_k))

# —— Batch processing multiple queries concurrently —— #
async def batch_answer_queries(queries: List[str], top_k: int = 3) -> List[str]:
    """并发处理多个查询"""
    tasks = [answer_query_async(query, top_k) for query in queries]
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
    """异步命令行界面"""
    print("简历问答系统 (输入 'quit' 退出)")
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("正在查询...")
        try:
            result = await timed_answer_query(query)
            print("\n回答:")
            print(result["answer"])
            print("\n性能指标:")
            metrics = result["metrics"]
            print(f"- 检索时间: {metrics['retrieval_time']:.3f}秒")
            print(f"- 提示构建: {metrics['prompt_time']:.3f}秒")
            print(f"- API调用: {metrics['api_time']:.3f}秒")
            print(f"- 总时间: {metrics['total_time']:.3f}秒")
        except Exception as e:
            print(f"错误: {e}")

# —— Local test —— #
if __name__ == "__main__":
    if asyncio.run(asyncio.sleep(0)) is None:  # 检查是否运行在支持异步的环境
        print("使用异步模式运行...")
        asyncio.run(run_cli())
    else:
        print("环境不支持异步，使用同步模式...")
        sample_query = "What are the highlights of my project experience?"
        print(">>", answer_query(sample_query))