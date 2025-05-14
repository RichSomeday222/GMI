import os
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except (ImportError, LookupError):
    nltk = None


def load_text(file_path: Path) -> str:
    """
    读取 PDF / TXT / MD 文件，返回纯文本。
    """
    ext = file_path.suffix.lower()
    if ext == '.pdf':
        if PdfReader is None:
            raise RuntimeError("请先 pip install PyPDF2，再重试")
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    elif ext in {'.txt', '.md'}:
        return file_path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


def semantic_split(
    text: str,
    threshold: float = 0.75,
    max_chunk_chars: int = 500
) -> list[str]:
    """
    语义感知切分：根据相邻句向量相似度决定段落边界。
    相似度高于阈值时合并，否则开启新块。
    """
    # 1. 简单分句（中文按句号，也可替换为 nltk 方式）
    if nltk:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    else:
        sentences = re.split(r'(?<=[。！？\.\?!])\s*', text)
        sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return []

    # 2. 加载和缓存模型
    global _semantic_model
    try:
        _semantic_model
    except NameError:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. 句向量编码
    embeddings = _semantic_model.encode(sentences, convert_to_numpy=True)

    # 4. 按相似度合并句子
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0])
        sentence_len = len(sentences[i])
        if sim >= threshold and current_len + sentence_len <= max_chunk_chars:
            current_chunk.append(sentences[i])
            current_len += sentence_len
        else:
            chunks.append("".join(current_chunk))
            current_chunk = [sentences[i]]
            current_len = sentence_len
    # 添加最后一块
    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks


def main():
    # 路径设置
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # 读取简历文本
    resume_file = data_dir / "resume.pdf"  # 或 .txt/.md
    text = load_text(resume_file)

    # 语义感知切分
    chunks = semantic_split(text, threshold=0.65, max_chunk_chars=500)

    # 写入文件
    for idx, chunk in enumerate(chunks, start=1):
        path = chunks_dir / f"chunk_{idx:03d}.txt"
        path.write_text(chunk, encoding='utf-8')
        print(f"[写入] {path.name} ({len(chunk)} 字符)")


if __name__ == "__main__":
    main()
