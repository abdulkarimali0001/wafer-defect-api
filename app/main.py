"""REST API that serves the wafer defect classifier.

Run locally:   uvicorn app.main:app --reload
Docs (auto):   http://localhost:8000/docs

Endpoints
    GET  /health          is the service up and the model loaded?
    GET  /model-info      classes, parameters, threshold
    POST /predict         one wafer map (52x52 grid of 0/1/2) -> defect labels + probabilities
    POST /predict/batch   up to 256 wafer maps in one call
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.model import WaferNet, count_params

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"]
SIZE = 52
THRESHOLD = float(os.getenv("WAFER_THRESHOLD", "0.5"))
MODEL_PATH = Path(os.getenv("WAFER_MODEL", Path(__file__).resolve().parent.parent / "model" / "wafernet.pt"))
MAX_BATCH = 256

torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
model = WaferNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

app = FastAPI(title="Wafer Defect Classifier API", version="1.0.0",
              description="Classifies mixed-type defect patterns on 52x52 semiconductor wafer maps.")


class WaferMap(BaseModel):
    wafer_map: list[list[int]] = Field(..., description="52x52 grid: 0 = no die, 1 = pass, 2 = fail")

    @field_validator("wafer_map")
    @classmethod
    def check_shape(cls, v):
        if len(v) != SIZE or any(len(row) != SIZE for row in v):
            raise ValueError(f"wafer_map must be {SIZE}x{SIZE}")
        if any(x not in (0, 1, 2) for row in v for x in row):
            raise ValueError("values must be 0 (no die), 1 (pass) or 2 (fail)")
        return v


class Batch(BaseModel):
    wafers: list[WaferMap]


class Prediction(BaseModel):
    pattern: str                      # e.g. "Center+Scratch" or "Normal"
    defects: list[str]
    probabilities: dict[str, float]
    failed_die_ratio: float           # share of dies on the wafer that failed
    latency_ms: float


def encode(grids: list[list[list[int]]]) -> torch.Tensor:
    a = np.asarray(grids, dtype=np.uint8)
    return torch.from_numpy(np.stack([(a == v) for v in (0, 1, 2)], axis=1).astype(np.float32))


def run(grids):
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = torch.sigmoid(model(encode(grids))).numpy()
    ms = (time.perf_counter() - t0) * 1000 / len(grids)
    out = []
    for g, p in zip(grids, probs):
        a = np.asarray(g)
        defects = [c for c, v in zip(CLASSES, p) if v >= THRESHOLD]
        dies = int((a > 0).sum())
        out.append(Prediction(pattern="+".join(defects) or "Normal", defects=defects,
                              probabilities={c: round(float(v), 4) for c, v in zip(CLASSES, p)},
                              failed_die_ratio=round(float((a == 2).sum() / dies), 4) if dies else 0.0,
                              latency_ms=round(ms, 3)))
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {"model": "WaferNet CNN", "classes": CLASSES, "parameters": count_params(model),
            "input": f"{SIZE}x{SIZE} grid of 0/1/2", "threshold": THRESHOLD,
            "test_macro_f1": 0.991, "test_exact_match": 0.985}


@app.post("/predict", response_model=Prediction)
def predict(w: WaferMap):
    return run([w.wafer_map])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(b: Batch):
    if not 1 <= len(b.wafers) <= MAX_BATCH:
        raise HTTPException(422, f"send between 1 and {MAX_BATCH} wafers")
    return run([w.wafer_map for w in b.wafers])
