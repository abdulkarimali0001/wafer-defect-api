# Wafer Defect Classifier API

Production-style REST service for the [wafer map defect classifier](../wafer-defect-classifier) (test macro-F1 0.991): **FastAPI + Docker + automated tests + CI**. It turns a trained notebook model into something a fab's inspection system could call.

> 웨이퍼 맵 불량 분류 모델을 FastAPI 기반 REST API로 배포한 프로젝트입니다. 입력 검증, 배치 추론, 자동화 테스트(pytest 10개), Docker 이미지, GitHub Actions CI를 갖췄으며, 단일 요청 지연시간은 중앙값 약 10ms입니다.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Service and model status (used by the Docker health check) |
| GET | `/model-info` | Classes, parameter count, threshold, test scores |
| POST | `/predict` | One 52×52 wafer map → pattern, defect list, probabilities, failed-die ratio |
| POST | `/predict/batch` | Up to 256 wafer maps in one call |

Interactive docs are generated automatically at `/docs`.

**Example**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d @<(python -c "import json;print(json.dumps({'wafer_map': json.load(open('examples/sample_wafers.json'))[0]['wafer_map']}))")
```

```json
{
  "pattern": "Center",
  "defects": ["Center"],
  "probabilities": {"Center": 1.0, "Donut": 0.0001, "Edge_Loc": 0.0019, "Edge_Ring": 0.0005,
                    "Loc": 0.0028, "Near_Full": 0.0004, "Scratch": 0.0036, "Random": 0.0003},
  "failed_die_ratio": 0.2654,
  "latency_ms": 10.9
}
```

## Engineering choices

| Concern | How it's handled |
| --- | --- |
| Bad input | Pydantic validation: exactly 52×52, values only 0/1/2, batch size 1–256. Bad requests get HTTP 422 with a clear message, never a crash |
| Correctness | Tests send real held-out wafers (Center, Scratch, Donut+Edge_Ring+Scratch, Normal) and check the predicted pattern |
| Speed | Model loaded once at startup, `torch.inference_mode()`, batch endpoint for throughput |
| Configuration | Threshold, model path, and thread count set by environment variables |
| Image size | CPU-only PyTorch wheel in the Dockerfile |
| Quality gate | GitHub Actions runs lint (ruff) and tests on every push, then builds the Docker image and smoke-tests `/health` |

## Results

| Measure | Value |
| --- | --- |
| Tests | 10 passed |
| Single request latency (end to end, incl. JSON validation) | p50 9.9 ms, p95 13.5 ms |
| Batch throughput (64 wafers per call) | about 315 wafers/second |

Measured on a 2-core CPU with `python benchmark.py`; numbers are in `results/benchmark.json`.

## How to run

```bash
# Local
pip install -r requirements-dev.txt
uvicorn app.main:app --reload          # open http://localhost:8000/docs
pytest -q                              # run the tests
python benchmark.py                    # latency and throughput

# Docker
docker build -t wafer-defect-api .
docker run -p 8000:8000 wafer-defect-api
```

## Project structure

```
app/main.py              FastAPI app: validation, inference, endpoints
app/model.py             WaferNet architecture (same as the training project)
model/wafernet.pt        trained weights (4.7 MB)
tests/test_api.py        10 tests: health, correctness on real wafers, validation errors
benchmark.py             latency and throughput measurement
Dockerfile               CPU-only serving image
.github/workflows/ci.yml lint, test, Docker build and smoke test
```

## Next steps

- Serve the quantized ONNX model from [wafer-model-optimization](../wafer-model-optimization) for lower latency.
- Add request logging and Prometheus metrics for monitoring.
