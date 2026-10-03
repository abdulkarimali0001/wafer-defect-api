"""API tests. Run with:  pytest -q"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLES = json.loads((Path(__file__).resolve().parent.parent / "examples" / "sample_wafers.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info():
    info = client.get("/model-info").json()
    assert len(info["classes"]) == 8 and info["parameters"] > 0


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["expected"] for s in SAMPLES])
def test_known_wafers_are_classified_correctly(sample):
    """Real test-set wafers (never seen in training) must get the right pattern."""
    r = client.post("/predict", json={"wafer_map": sample["wafer_map"]})
    assert r.status_code == 200
    body = r.json()
    assert body["pattern"] == sample["expected"]
    assert set(body["probabilities"]) == {"Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"}
    assert 0 <= body["failed_die_ratio"] <= 1


def test_batch_matches_single():
    wafers = [{"wafer_map": s["wafer_map"]} for s in SAMPLES]
    batch = client.post("/predict/batch", json={"wafers": wafers}).json()
    assert [p["pattern"] for p in batch] == [s["expected"] for s in SAMPLES]


def test_rejects_wrong_shape():
    r = client.post("/predict", json={"wafer_map": [[0] * 10] * 10})
    assert r.status_code == 422


def test_rejects_invalid_values():
    grid = [[1] * 52 for _ in range(52)]
    grid[0][0] = 7
    assert client.post("/predict", json={"wafer_map": grid}).status_code == 422


def test_rejects_empty_batch():
    assert client.post("/predict/batch", json={"wafers": []}).status_code == 422
