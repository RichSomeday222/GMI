# src/embed.py

import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    # 1. 配置路径
    project_root = Path(__file__).parent.parent
    chunks_dir   = project_root / "data" / "chunks"
    out_dir      = project_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载模型
    #    推荐 all-MiniLM-L6-v2：小巧、速度快、效果不错
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. 读取所有片段文本
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
    texts = [f.read_text(encoding="utf-8") for f in chunk_files]
    print(f"找到 {len(texts)} 个文本片段，开始生成嵌入…")

    # 4. 批量编码（convert_to_numpy=True 直接返回 NumPy 数组）
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("嵌入生成完毕，维度：", embeddings.shape)

    # 5. 保存结果
    vec_path  = out_dir / "resume_embeddings.npy"
    txt_path  = out_dir / "texts.json"

    # 保存向量矩阵（N × D）
    np.save(vec_path, embeddings)
    # 保存对应的原文列表
    with open(txt_path, "w", encoding="utf-8") as fp:
        json.dump(texts, fp, ensure_ascii=False, indent=2)

    print(f"向量已保存到：{vec_path}")
    print(f"文本映射已保存到：{txt_path}")

if __name__ == "__main__":
    main()
