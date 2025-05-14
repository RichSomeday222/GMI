
import json
import numpy as np
from pathlib import Path
from query_engine import retrieve, answer_query
from sklearn.metrics import precision_score
ROOT = Path(__file__).parent.parent
with open(ROOT / "embeddings" / "texts.json", "r", encoding="utf-8") as f:
    source_texts = json.load(f)

def load_test_set(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def eval_retrieval(test_set, top_k=3):
    precisions = []
    for case in test_set:
        retrieved = retrieve(case["query"], top_k)
        # 找到它们在 source_texts 里的下标
        indices = []
        for chunk in retrieved:
            try:
                idx = source_texts.index(chunk)
            except ValueError:
                idx = None
            indices.append(idx)

        # 打印对比
        print(f"\nQuery: {case['query']}")
        print(f"  Labeled relevant: {case['relevant_chunks']}")
        print(f"  Retrieved indices: {indices}")
        for i, chunk in zip(indices, retrieved):
            print(f"    [{i}] {chunk[:60]}...")

        # 计算 P@k
        hits = len(set([i for i in indices if i is not None]) & set(case["relevant_chunks"]))
        precisions.append(hits / top_k)

    return np.mean(precisions)


def simple_answer_match(generated, expected):
    # 最简单的字面匹配比例
    return float(len(set(generated.split()) & set(expected.split())) / len(expected.split()))

def eval_generation(test_set):
    scores = []
    for case in test_set:
        answer = answer_query(case["query"])
        score  = simple_answer_match(answer, case["expected_answer"])
        scores.append(score)
    return np.mean(scores)

if __name__ == "__main__":
    test_path = Path(__file__).parent.parent / "data" / "test_set.json"
    test_set  = load_test_set(test_path)
    
    print("=== 检索评估 ===")
    prec = eval_retrieval(test_set, top_k=3)
    print(f"平均检索精确率 (P@3): {prec:.2f}")

    print("\n=== 生成评估 ===")
    gen_score = eval_generation(test_set)
    print(f"平均回答质量 (简单匹配得分): {gen_score:.2f}")
