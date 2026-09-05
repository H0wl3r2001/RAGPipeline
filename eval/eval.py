"""Evaluation harness: runs test_set.json against the running RAG API.

Scores retrieval hit-rate (expected source in returned sources) and
answer relevance (fraction of expected keywords found in the answer).
Prints a per-question table and aggregate scores. Supports flag-driven
subset selection (--limit / --ids), prompt-variant selection (--variant),
and a full sweep across all variants in app/prompts/ (--sweep).

Run via ephemeral container (see PROJECT_BRIEF.md section 5):
  docker run --rm --network ragpipeline_default \
    -v "$(pwd)/eval:/eval" -w /eval \
    -v eval_pip_cache:/root/.cache/pip \
    python:3.12-slim \
    bash -c "pip install -q -r requirements.txt && python eval.py"
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_URL = "http://app:8000"
TEST_SET = Path(__file__).parent / "test_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
QUERY_TIMEOUT = 600.0


def load_test_set() -> list[dict]:
    with open(TEST_SET, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        sys.exit("test_set.json is empty")
    return data


def select_subset(test_set: list[dict], limit: int | None, ids: list[str] | None) -> list[dict]:
    if limit is None and ids is None:
        return test_set
    if limit is not None and ids is not None:
        sys.exit("Use either --limit or --ids, not both.")
    if limit is not None:
        if limit < 1:
            sys.exit("--limit must be >= 1")
        return test_set[:limit]
    if ids is not None:
        if not ids:
            sys.exit("--ids must list at least one id")
        by_id = {item["id"]: item for item in test_set}
        missing = [qid for qid in ids if qid not in by_id]
        if missing:
            sys.exit(f"Unknown test ids: {', '.join(missing)}")
        original_order = {id(item): i for i, item in enumerate(test_set)}
        return sorted([by_id[qid] for qid in ids], key=lambda it: original_order[id(it)])


def query_api(question: str, variant: str | None = None) -> dict:
    payload: dict = {"question": question}
    if variant is not None:
        payload["prompt_variant"] = variant
    resp = httpx.post(
        f"{API_URL}/query",
        json=payload,
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


@dataclasses.dataclass
class RowResult:
    qid: str
    retrieval_ok: bool
    relevance: float
    latency_s: float
    answer: str


def run_one(item: dict, variant: str | None) -> RowResult:
    qid = item["id"]
    question = item["question"]
    expected_source = item["expected_source"]
    expected_keywords = item.get("expected_keywords", [])

    t0 = time.time()
    try:
        result = query_api(question, variant=variant)
    except Exception as e:
        latency = time.time() - t0
        print(f"  {qid:<10}  ERROR ({e.__class__.__name__}) latency={latency:.1f}s")
        return RowResult(qid=qid, retrieval_ok=False, relevance=0.0, latency_s=latency, answer="")
    latency = time.time() - t0

    answer = result.get("answer", "")
    retrieval_ok = check_retrieval(result, expected_source)
    relevance = check_relevance(answer, expected_keywords)
    return RowResult(
        qid=qid, retrieval_ok=retrieval_ok, relevance=relevance,
        latency_s=latency, answer=answer,
    )


def print_rows(rows: list[RowResult]) -> None:
    headers = ["ID", "Retrieval", "Relevance", "Score", "Latency", "Answer (truncated)"]
    widths = [10, 12, 12, 8, 10, 50]
    sep = "-" * (sum(widths) + len(widths) * 3)
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for r in rows:
        answer_short = (r.answer[: widths[5]] + "...") if len(r.answer) > widths[5] else r.answer
        row = [
            r.qid.ljust(widths[0]),
            ("HIT" if r.retrieval_ok else "MISS").ljust(widths[1]),
            f"{r.relevance:.0%}".ljust(widths[2]),
            f"{r.relevance:.2f}".ljust(widths[3]),
            f"{r.latency_s:.1f}s".ljust(widths[4]),
            answer_short.replace("\n", " ").ljust(widths[5]),
        ]
        print("  ".join(row))
    print(sep)


def aggregate(rows: list[RowResult]) -> dict:
    n = len(rows) or 1
    hits = sum(1 for r in rows if r.retrieval_ok)
    avg_rel = sum(r.relevance for r in rows) / n
    avg_lat = sum(r.latency_s for r in rows) / n
    return {
        "n": len(rows),
        "hit_rate": hits / n if rows else 0.0,
        "avg_relevance": avg_rel,
        "avg_latency_s": avg_lat,
        "hits": hits,
    }


def fmt_aggregate(agg: dict) -> str:
    return (
        f"hit {agg['hits']}/{agg['n']} ({agg['hit_rate']:.0%})"
        f"  rel {agg['avg_relevance']:.0%}"
        f"  latency {agg['avg_latency_s']:.1f}s"
    )


def run_one_variant(test_set: list[dict], total_loaded: int, variant: str | None) -> list[RowResult]:
    label = variant or "(default from .env)"
    print(f"\n=== Variant: {label} ===")
    rows: list[RowResult] = []
    for item in test_set:
        rows.append(run_one(item, variant=variant))
    print_rows(rows)
    agg = aggregate(rows)
    scope = "full" if len(rows) == total_loaded else "partial - use full set before recording results"
    print(f"  Ran {len(rows)}/{total_loaded} questions ({scope})")
    print(f"  {fmt_aggregate(agg)}")
    return rows


def list_api_variants() -> list[str]:
    resp = httpx.get(f"{API_URL}/prompts", timeout=10.0)
    resp.raise_for_status()
    return resp.json().get("variants", [])


def write_history(
    rows_by_variant: dict[str | None, list[RowResult]],
    total_loaded: int,
    test_set_subset: list[dict],
    sweep: bool,
) -> Path | None:
    if not rows_by_variant:
        return None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    variants_payload = []
    for variant, rows in rows_by_variant.items():
        agg = aggregate(rows)
        variants_payload.append(
            {
                "name": variant,
                **agg,
                "questions": [dataclasses.asdict(r) for r in rows],
            }
        )
    payload = {
        "timestamp": ts,
        "mode": "sweep" if sweep else "single",
        "subset_ids": (
            [item["id"] for item in test_set_subset]
            if sweep or len(test_set_subset) < total_loaded
            else None
        ),
        "variants": variants_payload,
    }
    out = RESULTS_DIR / f"{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n(wrote {out})")
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG eval harness against the API. Defaults to the full test set."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--limit",
        type=int,
        help="Run only the first N questions (file order).",
    )
    group.add_argument(
        "--ids",
        type=str,
        help="Comma-separated list of question IDs to run (file order).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Prompt template variant to send to /query (Phase 6). Defaults to PROMPT_VARIANT from .env.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the test set once per variant advertised by GET /prompts and print a comparison table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    full_set = load_test_set()
    ids_arg = [s.strip() for s in args.ids.split(",")] if args.ids else None
    subset = select_subset(full_set, args.limit, ids_arg)

    start = time.time()
    rows_by_variant: dict[str | None, list[RowResult]] = {}
    if args.sweep:
        variants = list_api_variants()
        print(f"Sweep mode: variants advertised by API = {variants}")
        for v in variants:
            rows_by_variant[v] = run_one_variant(subset, total_loaded=len(full_set), variant=v)
        # Comparison summary
        print("\n=== Comparison summary ===")
        hdr = ["variant", "hits", "relevance", "avg_lat_s"]
        widths2 = [22, 10, 12, 12]
        print("  ".join(h.ljust(w) for h, w in zip(hdr, widths2)))
        for v, rows in rows_by_variant.items():
            a = aggregate(rows)
            print("  ".join([
                (v or "(default)").ljust(widths2[0]),
                f"{a['hits']}/{a['n']}".ljust(widths2[1]),
                f"{a['avg_relevance']:.0%}".ljust(widths2[2]),
                f"{a['avg_latency_s']:.1f}".ljust(widths2[3]),
            ]))
    else:
        rows = run_one_variant(subset, total_loaded=len(full_set), variant=args.variant)
        rows_by_variant[args.variant] = rows

    write_history(
        rows_by_variant=rows_by_variant,
        total_loaded=len(full_set),
        test_set_subset=subset,
        sweep=args.sweep,
    )
    print(f"\n(completed in {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
