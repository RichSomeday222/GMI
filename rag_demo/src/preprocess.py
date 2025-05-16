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
            raise RuntimeError("Please run 'pip install PyPDF2' and try again")
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    elif ext in {'.txt', '.md'}:
        return file_path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {ext}")



def semantic_split(
    text: str,
    threshold: float = 0.8,
    max_chunk_chars: int = 800,
    min_chunk_chars: int = 200
) -> list[str]:
    """
    语义感知切分：根据相邻句向量相似度决定段落边界。
    相似度高于阈值时合并，否则开启新块；对过短的 chunk 进行二次合并。
    Semantic-aware segmentation: Determine paragraph boundaries based on 
    the similarity of adjacent sentence vectors.
    Merge when the similarity is higher than the threshold, otherwise start 
    a new block; merge too short chunks twice.
    """
    # 1. 简单分句
    if nltk:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    else:
        sentences = re.split(r'(?<=[。！？\.\?!])\s*', text)
        sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return []

    # 2. 缓存模型
    global _semantic_model
    try:
        _semantic_model
    except NameError:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. 句向量编码
    embeddings = _semantic_model.encode(sentences, convert_to_numpy=True)

    # 4. 语义边界切分
    raw_chunks = []
    curr_chunk = [sentences[0]]
    curr_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0])
        sent_len = len(sentences[i])
        if sim >= threshold and curr_len + sent_len <= max_chunk_chars:
            curr_chunk.append(sentences[i])
            curr_len += sent_len
        else:
            raw_chunks.append("".join(curr_chunk))
            curr_chunk = [sentences[i]]
            curr_len = sent_len
    if curr_chunk:
        raw_chunks.append("".join(curr_chunk))

    # 5. 合并过短 chunk
    final_chunks = []
    for chunk in raw_chunks:
        if final_chunks and len(chunk) < min_chunk_chars:
            final_chunks[-1] += chunk
        else:
            final_chunks.append(chunk)
    return final_chunks


def main():
    # 路径设置
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # 读取简历文本
    resume_file = data_dir / "resume.pdf"
    text = load_text(resume_file)

    # 语义感知切分，
    chunks = semantic_split(text, threshold=0.8, max_chunk_chars=800, min_chunk_chars=200)

    # 写入
    for idx, chunk in enumerate(chunks, start=1):
        path = chunks_dir / f"chunk_{idx:03d}.txt"
        path.write_text(chunk, encoding='utf-8')
        print(f"[Write] {path.name} ({len(chunk)} Characters)")


if __name__ == "__main__":
    main()
