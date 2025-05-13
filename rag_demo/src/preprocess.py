# src/preprocess.py

import os
import re
from pathlib import Path

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

def simple_chunks(
    text: str,
    max_sentences: int = 5,
    max_chars: int = 500
) -> list[str]:
    """
    按句子数或字符数切分：先分句，再累积到阈值就生成一个 chunk。
    """
    # 1. 分句
    if nltk:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    else:
        sents = re.split(r'(?<=[。！？\.!\?])\s*', text)
        sents = [s.strip() for s in sents if s.strip()]

    # 2. 累积切块
    chunks, curr, curr_len = [], [], 0
    for s in sents:
        if len(curr) >= max_sentences or curr_len + len(s) > max_chars:
            chunks.append("".join(curr))
            curr, curr_len = [], 0
        curr.append(s)
        curr_len += len(s)
    if curr:
        chunks.append("".join(curr))
    return chunks

def sliding_windows(
    text: str,
    max_chars: int = 512,
    stride: int = 1
) -> list[str]:
    """
    基于滑窗对单条长文本做重叠切分，每次滑动 stride 个句子。
    """
    # 1. 分句
    sents = text.split("。")
    sents = [s for s in sents if s.strip()]
    windows = []
    n = len(sents)
    for start in range(0, n, stride):
        chunk = "".join(sents[start:start + max(1, max_chars // 20)])
        # 若 chunk 真正长度超 max_chars，再截断
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars]
        windows.append(chunk)
        if start + max_chars // 20 >= n:
            break
    return windows

def hybrid_split(
    text: str,
    max_sentences: int = 5,
    simple_max_chars: int = 500,
    window_max_chars: int = 512,
    stride: int = 1,
    long_thresh: int = 600
) -> list[str]:
    """
    混合切分：先做简单分句阈值切分，
    再对“超长”块用滑窗二次细分。
    """
    chunks = simple_chunks(text, max_sentences, simple_max_chars)
    out_chunks = []
    for c in chunks:
        if len(c) > long_thresh:
            # 对长 chunk 做滑窗切分
            windows = sliding_windows(c, window_max_chars, stride)
            out_chunks.extend(windows)
        else:
            out_chunks.append(c)
    return out_chunks

def main():
    # 路径设置
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # 读取简历文本
    resume_file = data_dir / "resume.pdf"  # 或 .txt/.md
    text = load_text(resume_file)

    # 切分
    chunks = hybrid_split(
        text,
        max_sentences=5,
        simple_max_chars=500,
        window_max_chars=512,
        stride=1,
        long_thresh=600
    )
    # 写入
    for idx, chunk in enumerate(chunks, 1):
        path = chunks_dir / f"chunk_{idx:03d}.txt"
        path.write_text(chunk, encoding='utf-8')
        print(f"[写入] {path.name} （{len(chunk)} 字符）")

if __name__ == "__main__":
    main()
