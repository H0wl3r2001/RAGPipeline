"""Evaluation harness: runs test_set.json against the running RAG API.

Scores retrieval hit-rate (expected source in returned sources) and
answer relevance (fraction of expected keywords found in the answer).
Prints a per-question table and aggregate scores.

Run via ephemeral container (see PROJECT_BRIEF.md section 5):
  docker run --rm --network ragpipeline_default \
    -v "$(pwd)/eval:/eval" -w /eval \
    -v eval_pip_cache:/root/.cache/pip \
    python:3.12-slim \
    bash -c "pip install -q -r requirements.txt && python eval.py"
"""

import json
import sys
import time
from pathlib import Path

import httpx

API_URL = "http://app:8000"
TEST_SET = Path(__file__).parent / "test_set.json"
QUERY_TIMEOUT = 600.0


def load_test_set() -> list[dict]:
    with open(TEST_SET, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        sys.exit("test_set.json is empty")
    return data


def query_api(question: str) -> dict:
    resp = httpx.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=QUERY_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def check_retrieval(result: dict, expected_source: str) -> bool:
    return any(s["file"] == expected_source for s in result.get("sources", []))


def check_relevance(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_eval() -> None:
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test cases from test_set.json\n")

    headers = ["ID", "Retrieval", "Relevance", "Score", "Answer (truncated)"]
    widths = [10, 12, 12, 8, 50]

    print("-" * (sum(widths) + len(widths) * 3 + 1))
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + len(widths) * 3 + 1))

    retrieval_hits = 0
    relevance_scores: list[float] = []

    for item in test_set:
        qid = item["id"]
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item.get("expected_keywords", [])

        try:
            result = query_api(question)
        except Exception as e:
            err = str(e)[:60]
            print(f"  {qid:<{widths[0]}}  {'ERROR':<{widths[1]}}  {'':<{widths[2]}}  {'':<{widths[3]}}  {err}")
            relevance_scores.append(0.0)
            continue

        answer = result.get("answer", "")
        retrieval_ok = check_retrieval(result, expected_source)
        relevance = check_relevance(answer, expected_keywords)

        if retrieval_ok:
            retrieval_hits += 1
        relevance_scores.append(relevance)

        answer_short = answer[:widths[4]] + ("..." if len(answer) > widths[4] else "")
        row = [
            qid.ljust(widths[0]),
            ("HIT" if retrieval_ok else "MISS").ljust(widths[1]),
            f"{relevance:.0%}".ljust(widths[2]),
            f"{relevance:.2f}".ljust(widths[3]),
            answer_short.replace("\n", " ").ljust(widths[4]),
        ]
        print("  ".join(row))

    print("-" * (sum(widths) + len(widths) * 3 + 1))

    total = len(test_set)
    hit_rate = retrieval_hits / total if total else 0.0
    avg_relevance = sum(relevance_scores) / total if total else 0.0

    print()
    print(f"  Retrieval hit-rate : {retrieval_hits}/{total} = {hit_rate:.0%}")
    print(f"  Avg answer relevance: {avg_relevance:.0%}")
    print()


if __name__ == "__main__":
    start = time.time()
    run_eval()
    print(f"(completed in {time.time() - start:.1f}s)")
