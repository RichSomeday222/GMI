# src/build_index.py

import os
import numpy as np
import faiss
from pathlib import Path

def main():
    # 1. 路径配置
    project_root = Path(__file__).parent.parent
    emb_dir      = project_root / "embeddings"
    idx_dir      = project_root / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / "resume_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"找不到嵌入文件: {emb_path}")

    # 2. 加载嵌入向量
    embeddings = np.load(str(emb_path))
    # embeddings.shape = (N, D)
    n, dim = embeddings.shape
    print(f"加载 {n} 个向量，维度={dim}")

    # 3. （可选）向量归一化，以便后面使用内积等价于余弦相似度
    faiss.normalize_L2(embeddings)

    # 4. 构建索引：IndexFlatIP 使用内积 (inner product)
    index = faiss.IndexFlatIP(dim)

    # 5. 添加向量到索引
    index.add(embeddings)
    print(f"向索引中添加了 {index.ntotal} 个向量")

    # 6. 持久化索引到磁盘
    index_path = idx_dir / "resume.index"
    faiss.write_index(index, str(index_path))
    print(f"FAISS 索引已保存到: {index_path}")

if __name__ == "__main__":
    main()
