"""Basic smoke tests for the RAG API.

Run via ephemeral container (see PROJECT_BRIEF.md section 5):
  docker run --rm --network ragpipeline_default \
    -v "$(pwd)/tests:/tests" -w /tests \
    -v eval_pip_cache:/root/.cache/pip \
    python:3.12-slim \
    bash -c "pip install -q -r requirements.txt && python test_api.py"
"""

import sys

import httpx

API_URL = "http://app:8000"
TIMEOUT = 600.0


def test_health():
    resp = httpx.get(f"{API_URL}/health", timeout=10.0)
    assert resp.status_code == 200, f"health status {resp.status_code}"
    data = resp.json()
    assert data["app"] == "ok", f"app not ok: {data}"
    print("PASS: /health returns app=ok")


def test_ingest():
    resp = httpx.post(f"{API_URL}/ingest", timeout=600.0)
    assert resp.status_code == 200, f"ingest status {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["status"] == "ok", f"ingest status field: {data}"
    assert data["chunks"] > 0, f"no chunks ingested: {data}"
    print(f"PASS: /ingest returned {data['chunks']} chunks")


def test_query():
    resp = httpx.post(
        f"{API_URL}/query",
        json={"question": "What is the global economic growth outlook?"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, f"query status {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "answer" in data, f"no answer field: {data}"
    assert "sources" in data, f"no sources field: {data}"
    assert isinstance(data["sources"], list), f"sources not a list: {data}"
    print(f"PASS: /query returned answer ({len(data['answer'])} chars) with {len(data['sources'])} sources")


def test_query_empty():
    resp = httpx.post(
        f"{API_URL}/query",
        json={"question": "   "},
        timeout=10.0,
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}"
    print("PASS: /query empty question returns 400")


if __name__ == "__main__":
    tests = [test_health, test_ingest, test_query, test_query_empty]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
