# docs/SERVING.md — API, Inference Pipeline & Deployment

> Read before implementing: training/scripts/serving/api.py, inference_worker.py

---

## FastAPI Application Structure

```python
# training/scripts/serving/api.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, time

app = FastAPI(title="StreamGuard API", version="1.0.0")

# ── Request/Response Models ───────────────────────────────────────

class ScanFunctionRequest(BaseModel):
    code: str                        # C function source code
    context_callees: dict = {}       # {func_name: source_code} for inter-proc
    language: str = "c"

class TaintNode(BaseModel):
    node: str
    role: str                        # SOURCE | PROPAGATION | SINK | SANITIZER
    line: int

class KeyNode(BaseModel):
    node: str
    importance: float
    line: int

class CounterfactualHint(BaseModel):
    description: str
    fix_pattern: str
    minimal_fix_lines: list[int]

class ScanPrediction(BaseModel):
    function_name: str
    is_vulnerable: bool
    confidence: float
    cwe: str
    cwe_name: str
    severity_score: float
    severity_label: str              # LOW | MEDIUM | HIGH | CRITICAL
    taint_path: list[TaintNode]
    key_nodes: list[KeyNode]
    counterfactual_hint: CounterfactualHint | None
    model_version: str
    inference_ms: int
    scan_id: str

# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/v1/scan/function", response_model=ScanPrediction)
async def scan_function(req: ScanFunctionRequest):
    start = time.perf_counter()
    scan_id = f"sg-{uuid.uuid4().hex[:8]}"
    result = await inference_worker.scan(req.code, req.context_callees)
    result["inference_ms"] = int((time.perf_counter() - start) * 1000)
    result["scan_id"] = scan_id
    return result

@app.post("/v1/scan/file")
async def scan_file(code: str):
    """Extract all functions from file, scan each."""
    from tree_sitter_utils import extract_functions
    functions = extract_functions(code)
    results = []
    for fn in functions:
        result = await inference_worker.scan(fn["code"])
        results.append(result)
    return {"functions_scanned": len(results), "results": results}

@app.post("/v1/scan/batch")
async def scan_batch(background_tasks: BackgroundTasks):
    """Async batch scan. Returns job_id."""
    job_id = str(uuid.uuid4())
    # background_tasks.add_task(process_batch, job_id, ...)
    return {"job_id": job_id, "status": "queued"}

@app.post("/v1/feedback")
async def submit_feedback(scan_id: str, correct_label: int, correct_cwe: str = None):
    """Human correction → stored → triggers CFA generation queue."""
    feedback_store.save(scan_id, correct_label, correct_cwe)
    cfa_queue.enqueue(scan_id)
    return {"acknowledged": True}

@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "model_version": inference_worker.model_version,
        "gpu_available": torch.cuda.is_available(),
        "uptime_seconds": int(time.time() - START_TIME),
    }
```

---

## Inference Pipeline (< 300ms P95 target)

> **MANDATORY:** Call `model.eval()` before any inference. GroupNorm layers and dropout
> behave differently in eval vs train mode. Single-function scan (batch_size=1) is the
> normal serving case — `model.eval()` ensures correct normalization statistics.

```
Input: C function string
  │
  ▼ ~50ms
Pre-processing
  ├── tree-sitter: function boundary validation
  ├── Joern subprocess (pre-warmed pool of 4): CPG construction
  ├── Python taint analyzer: TPG edges
  └── Callee lookup: Redis cache → LLM (Claude Haiku) if miss
  │
  ▼ ~100ms
Feature extraction
  ├── CodeBERT tokenize (512 tokens max)
  ├── CPG node embedding (CodeBERT, 64 tok/node, batch GPU)
  ├── 824-d feature vector construction
  └── PyG Data object assembly
  │
  ▼ ~30ms
Model inference (GPU)
  ├── CodeBERT forward pass → [CLS] 768-d
  ├── GGNN 3-layer propagation → graph 256-d
  ├── Cross-attention fusion → 1280-d
  └── 3 heads: binary + CWE + severity
  │
  ▼ ~20ms
Post-processing
  ├── Threshold (0.5) → is_vulnerable
  ├── CWE index → CWE name
  ├── Illuminati: per-node importance scores (GNN attention weights)
  └── CFExplainer: counterfactual hint generation
  │
  ▼
Prediction JSON → logged → returned to client
```

---

## Joern Subprocess Pool

```python
# Joern takes ~700ms cold start but ~50ms once warmed
# Use a pool of 4 pre-warmed Joern processes

import subprocess, queue, threading

class JoernPool:
    def __init__(self, pool_size=4):
        self.pool = queue.Queue()
        for _ in range(pool_size):
            proc = self._start_joern_process()
            self.pool.put(proc)
    
    def _start_joern_process(self):
        """Start Joern in interactive mode (stdin/stdout)."""
        import os
        return subprocess.Popen(
            [os.environ["JOERN_BIN"], "--nocolors"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    
    def get_cpg(self, c_code: str) -> dict:
        proc = self.pool.get()
        try:
            result = self._run_cpg_extraction(proc, c_code)
        except Exception:
            proc = self._start_joern_process()  # restart on failure
            result = {}
        finally:
            self.pool.put(proc)
        return result
```

---

## Deployment Configuration

```yaml
# docker-compose.yml sketch
services:
  api:
    build: docker/Dockerfile.serving
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=./checkpoints/best_model.pt
      - JOERN_BIN=/opt/joern/joern-cli/joern
    depends_on: [redis, postgres]
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  redis:
    image: redis:7-alpine
    # Callee summary cache: ~32GB for 1M cached summaries

  postgres:
    image: timescale/timescaledb:latest-pg15
    # Prediction log + feedback store
```

---

*docs/SERVING.md | StreamGuard v1.0 | March 2026*
