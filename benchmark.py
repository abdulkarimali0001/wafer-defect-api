"""Measure API latency and throughput with the in-process test client.

Usage:  python benchmark.py
Writes results/benchmark.json
"""
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
samples = json.loads(Path("examples/sample_wafers.json").read_text())
one = {"wafer_map": samples[0]["wafer_map"]}

for _ in range(20):  # warm-up
    client.post("/predict", json=one)

single = []
for _ in range(300):
    t0 = time.perf_counter(); client.post("/predict", json=one); single.append((time.perf_counter() - t0) * 1000)

batch = {"wafers": [{"wafer_map": samples[i % len(samples)]["wafer_map"]} for i in range(64)]}
t0 = time.perf_counter()
for _ in range(20):
    client.post("/predict/batch", json=batch)
batch_rate = 20 * 64 / (time.perf_counter() - t0)

single.sort()
res = {
    "single_request_ms": {"p50": round(statistics.median(single), 2), "p95": round(single[int(0.95 * len(single))], 2)},
    "batch64_wafers_per_second": round(batch_rate, 1),
    "note": "End-to-end incl. JSON parsing and validation, CPU, 2 PyTorch threads, in-process client.",
}
Path("results").mkdir(exist_ok=True)
Path("results/benchmark.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
