# 04 - Platform-Independent Local Agent Architecture

**Phase:** 3 (Weeks 7-8)  
**Prerequisites:** ML models trained ([02_ml_training.md](./02_ml_training.md)), Explainability system ([03_explainability.md](./03_explainability.md))  
**Status:** Ready to Implement

---

## 📋 Overview

Build a universal local detection agent that runs on developers' machines and integrates with any IDE through REST/WebSocket APIs and JSON-RPC protocol.

**Key Features:**
- **FastAPI Server**: REST + WebSocket + JSON-RPC endpoints
- **Cross-Platform File Monitor**: inotify (Linux), FSEvents (macOS), ReadDirectoryChangesW (Windows)
- **Incremental Analysis**: Diff-based detection with caching
- **Universal IDE Support**: Works with VS Code, IntelliJ, Vim, Emacs, any editor
- **Real-Time Updates**: WebSocket streaming
- **Docker Deployment**: Containerized with compose
- **Low Latency**: <100ms response time

**Deliverables:**
- ✅ FastAPI agent server with JSON-RPC
- ✅ Cross-platform file monitor
- ✅ Incremental diff engine
- ✅ WebSocket real-time streaming
- ✅ Docker deployment
- ✅ IDE integration examples

**Expected Time:** 2 weeks

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Universal IDE Integration                   │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │ VS Code    │  │ IntelliJ   │  │ Any IDE via              │  │
│  │ Extension  │  │ Plugin     │  │ JSON-RPC/REST/WebSocket  │  │
│  └──────┬─────┘  └──────┬─────┘  └──────────┬───────────────┘  │
│         │                │                    │                  │
│         └────────────────┴────────────────────┘                  │
└─────────────────────────┬──────────────────────────────────────┘
                          │ HTTP/WebSocket/JSON-RPC
                          │
┌─────────────────────────▼──────────────────────────────────────┐
│         StreamGuard Local Agent (localhost:8765)               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  FastAPI Server with Multiple Protocols                   │ │
│  │                                                            │ │
│  │  REST API:                                                 │ │
│  │  • POST /analyze          • GET /status                   │ │
│  │  • POST /feedback         • GET /patterns                 │ │
│  │  • GET /health            • DELETE /cache                 │ │
│  │                                                            │ │
│  │  WebSocket:                                                │ │
│  │  • ws://localhost:8765/stream                             │ │
│  │    - Real-time vulnerability updates                      │ │
│  │    - Analysis progress streaming                          │ │
│  │                                                            │ │
│  │  JSON-RPC 2.0:                                             │ │
│  │  • POST /rpc                                               │ │
│  │    - analyze(), getStatus(), submitFeedback()             │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Request Router & Middleware                              │ │
│  │  • CORS (localhost only)                                  │ │
│  │  • Rate limiting (100 req/min)                            │ │
│  │  • Authentication (API keys)                              │ │
│  │  • Request validation                                     │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  File System Monitor (Cross-Platform)                     │ │
│  │                                                            │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │ │
│  │  │  inotify   │  │  FSEvents  │  │ ReadDirectoryChanges│ │ │
│  │  │  (Linux)   │  │  (macOS)   │  │  (Windows)          │ │ │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │ │
│  │                                                            │ │
│  │  Features:                                                 │ │
│  │  • Watches project directories recursively                │ │
│  │  • Filters supported file types (.py, .js, .ts, etc.)    │ │
│  │  • Debounces rapid changes (500ms)                        │ │
│  │  • Ignores node_modules, .git, etc.                       │ │
│  │  • Async event handling                                   │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Incremental Diff Engine                                  │ │
│  │                                                            │ │
│  │  • Computes line-level diffs (difflib)                    │ │
│  │  • Identifies changed functions (AST diff)                │ │
│  │  • Extracts relevant context (±20 lines)                  │ │
│  │  • Caches previous versions (Redis/memory)                │ │
│  │  • Deduplicates identical changes                         │ │
│  │  • Smart context expansion                                │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Analysis Orchestrator                                     │ │
│  │                                                            │ │
│  │  Multi-Agent Pipeline:                                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Syntax Agent (50ms)                                  │ │ │
│  │  │ • Pattern matching                                   │ │ │
│  │  │ • AST-based detection                                │ │ │
│  │  └────────────┬─────────────────────────────────────────┘ │ │
│  │  ┌────────────▼─────────────────────────────────────────┐ │ │
│  │  │ Semantic Agent (200ms)                               │ │ │
│  │  │ • ML-based detection (GNN + Transformer)             │ │ │
│  │  │ • Taint flow analysis                                │ │ │
│  │  │ • Explainability (Integrated Gradients)              │ │ │
│  │  └────────────┬─────────────────────────────────────────┘ │ │
│  │  ┌────────────▼─────────────────────────────────────────┐ │ │
│  │  │ Context Agent (500ms)                                │ │ │
│  │  │ • Repository graph queries (Neo4j)                   │ │ │
│  │  │ • Cross-file taint propagation                       │ │ │
│  │  └────────────┬─────────────────────────────────────────┘ │ │
│  │  ┌────────────▼─────────────────────────────────────────┐ │ │
│  │  │ Verification Agent (300ms)                           │ │ │
│  │  │ • Query reconstruction                               │ │ │
│  │  │ • Attack simulation                                  │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Result Cache & State Manager                             │ │
│  │                                                            │ │
│  │  • Redis cache (optional)                                 │ │
│  │  • In-memory LRU cache (10,000 entries)                   │ │
│  │  • File checksums (SHA256)                                │ │
│  │  • WebSocket connection pool                              │ │
│  │  • Analysis history                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ Results + Explanations
                          ▼
                  IDE Display (Inline)
```

### Data Flow

```
File Change (save/commit)
        ↓
File System Monitor detects change
        ↓
Compute diff (only changed parts)
        ↓
Check cache (file hash)
        ↓
    Cache Hit? ──Yes──→ Return cached result (<10ms)
        │
        No
        ↓
Run Detection Pipeline
        ↓
Syntax (50ms) → Semantic (200ms) → Context (500ms) → Verification (300ms)
        ↓
Generate Explanation
        ↓
Cache result
        ↓
Stream to WebSocket clients
        ↓
Return response to IDE (<1s total)
```

---

## 💻 Implementation

### 1. FastAPI Server Core

**File:** `core/engine/local_agent.py`

```python
"""Platform-independent local detection agent with FastAPI."""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import uvicorn
import asyncio
from datetime import datetime
import hashlib
import json
from collections import deque
import time

app = FastAPI(
    title="StreamGuard Local Agent",
    description="Universal vulnerability detection agent",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - only localhost for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://0.0.0.0:*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ==================== Request/Response Models ====================

class AnalysisRequest(BaseModel):
    """Request for code analysis."""
    code: str = Field(..., min_length=1, description="Source code to analyze")
    language: str = Field(..., description="Programming language")
    file_path: Optional[str] = Field(None, description="File path for context")
    project_root: Optional[str] = Field(None, description="Project root directory")
    incremental: bool = Field(True, description="Use incremental analysis")
    previous_hash: Optional[str] = Field(None, description="Hash of previous version")
    
    @validator('language')
    def validate_language(cls, v):
        supported = ['python', 'javascript', 'typescript', 'java', 'go', 'php', 'ruby', 'sql']
        if v.lower() not in supported:
            raise ValueError(f"Unsupported language: {v}. Supported: {supported}")
        return v.lower()


class Vulnerability(BaseModel):
    """Detected vulnerability."""
    id: str
    type: str
    severity: str  # critical, high, medium, low
    confidence: float = Field(..., ge=0.0, le=1.0)
    line: int
    column: Optional[int] = None
    end_line: Optional[int] = None
    message: str
    explanation: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    cve_references: List[str] = []


class AnalysisResponse(BaseModel):
    """Response from analysis."""
    status: str
    analysis_time_ms: float
    vulnerabilities: List[Vulnerability]
    file_hash: str
    cached: bool = False
    incremental: bool = False
    stats: Dict[str, Any] = {}


class FeedbackRequest(BaseModel):
    """User feedback on a vulnerability."""
    vulnerability_id: str
    action: str = Field(..., regex="^(accepted|rejected|false_positive|helpful)$")
    comment: Optional[str] = None
    code_context: Optional[str] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None


# ==================== Global State ====================

class AgentState:
    """Agent state management with LRU cache."""
    
    def __init__(self, max_cache_size: int = 10000):
        self.active_connections: List[WebSocket] = []
        self.analysis_cache: Dict[str, AnalysisResponse] = {}
        self.cache_access_order: deque = deque(maxlen=max_cache_size)
        self.max_cache_size = max_cache_size
        self.file_checksums: Dict[str, str] = {}
        self.start_time = time.time()
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "average_time_ms": 0.0,
            "vulnerabilities_found": 0,
            "false_positives": 0
        }
        self.rate_limiter: Dict[str, List[float]] = {}  # IP -> timestamps
    
    def check_rate_limit(self, client_ip: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
        """Check if client exceeds rate limit."""
        now = time.time()
        
        if client_ip not in self.rate_limiter:
            self.rate_limiter[client_ip] = []
        
        # Remove old timestamps
        self.rate_limiter[client_ip] = [
            ts for ts in self.rate_limiter[client_ip]
            if now - ts < window_seconds
        ]
        
        # Check limit
        if len(self.rate_limiter[client_ip]) >= max_requests:
            return False
        
        self.rate_limiter[client_ip].append(now)
        return True
    
    def add_to_cache(self, key: str, value: AnalysisResponse):
        """Add to cache with LRU eviction."""
        if len(self.analysis_cache) >= self.max_cache_size:
            # Evict oldest
            oldest_key = self.cache_access_order.popleft()
            self.analysis_cache.pop(oldest_key, None)
        
        self.analysis_cache[key] = value
        self.cache_access_order.append(key)
    
    def get_from_cache(self, key: str) -> Optional[AnalysisResponse]:
        """Get from cache and update access order."""
        if key in self.analysis_cache:
            # Move to end (most recently used)
            try:
                self.cache_access_order.remove(key)
            except ValueError:
                pass
            self.cache_access_order.append(key)
            
            return self.analysis_cache[key]
        
        return None


state = AgentState()


# ==================== Middleware ====================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host
    
    # Skip rate limiting for health check
    if request.url.path in ["/", "/health", "/status"]:
        return await call_next(request)
    
    if not state.check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Max 100 requests per minute."}
        )
    
    response = await call_next(request)
    return response


# ==================== REST API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "StreamGuard Local Agent",
        "version": "3.0.0",
        "status": "running",
        "uptime_seconds": int(time.time() - state.start_time)
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - state.start_time),
        "cache_size": len(state.analysis_cache),
        "active_connections": len(state.active_connections),
        "stats": state.stats
    }


@app.get("/status")
async def get_status():
    """Get detailed agent status."""
    return {
        "status": "running",
        "uptime_seconds": int(time.time() - state.start_time),
        "stats": state.stats,
        "active_websocket_connections": len(state.active_connections),
        "cache": {
            "size": len(state.analysis_cache),
            "max_size": state.max_cache_size,
            "hit_rate": (
                state.stats["cache_hits"] / state.stats["total_analyses"]
                if state.stats["total_analyses"] > 0 else 0.0
            )
        },
        "performance": {
            "average_analysis_time_ms": state.stats["average_time_ms"],
            "total_analyses": state.stats["total_analyses"]
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze code for vulnerabilities.
    
    Main endpoint for IDE plugins. Supports incremental analysis with caching.
    """
    start_time = time.time()
    
    # Compute file hash
    file_hash = hashlib.sha256(request.code.encode('utf-8')).hexdigest()
    
    # Check cache if incremental mode
    if request.incremental:
        cached_result = state.get_from_cache(file_hash)
        if cached_result:
            cached_result.cached = True
            state.stats["cache_hits"] += 1
            
            # Notify WebSocket clients
            background_tasks.add_task(
                notify_websocket_clients,
                {
                    "type": "cache_hit",
                    "file_hash": file_hash,
                    "file_path": request.file_path
                }
            )
            
            return cached_result
    
    # Run detection pipeline
    try:
        vulnerabilities = await run_detection_pipeline(
            code=request.code,
            language=request.language,
            file_path=request.file_path,
            project_root=request.project_root,
            incremental=request.incremental,
            previous_hash=request.previous_hash
        )
        
        # Compute analysis time
        analysis_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = AnalysisResponse(
            status="success",
            analysis_time_ms=analysis_time_ms,
            vulnerabilities=vulnerabilities,
            file_hash=file_hash,
            cached=False,
            incremental=request.incremental,
            stats={
                "vulnerabilities_count": len(vulnerabilities),
                "critical_count": sum(1 for v in vulnerabilities if v.severity == "critical"),
                "high_count": sum(1 for v in vulnerabilities if v.severity == "high")
            }
        )
        
        # Update cache
        state.add_to_cache(file_hash, response)
        
        # Update stats
        state.stats["total_analyses"] += 1
        state.stats["vulnerabilities_found"] += len(vulnerabilities)
        state.stats["average_time_ms"] = (
            (state.stats["average_time_ms"] * (state.stats["total_analyses"] - 1) +
             analysis_time_ms) / state.stats["total_analyses"]
        )
        
        # Notify WebSocket clients
        background_tasks.add_task(
            notify_websocket_clients,
            {
                "type": "analysis_complete",
                "file_path": request.file_path,
                "vulnerabilities_count": len(vulnerabilities),
                "analysis_time_ms": analysis_time_ms,
                "vulnerabilities": [v.dict() for v in vulnerabilities]
            }
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on vulnerability detection."""
    try:
        # Store feedback locally
        from core.feedback.collector import FeedbackCollector
        
        collector = FeedbackCollector()
        await collector.add_feedback(
            vulnerability_id=request.vulnerability_id,
            action=request.action,
            comment=request.comment,
            code_context=request.code_context,
            timestamp=datetime.now()
        )
        
        # Update stats
        if request.action == "false_positive":
            state.stats["false_positives"] += 1
        
        return {
            "status": "success",
            "message": "Feedback recorded"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns")
async def get_patterns(project_root: Optional[str] = None):
    """Get detected vulnerability patterns for this project."""
    # TODO: Implement pattern retrieval from database
    return {
        "patterns": [],
        "count": 0,
        "project_root": project_root
    }


@app.delete("/cache")
async def clear_cache():
    """Clear analysis cache."""
    cache_size = len(state.analysis_cache)
    state.analysis_cache.clear()
    state.cache_access_order.clear()
    
    return {
        "status": "success",
        "message": f"Cleared {cache_size} cached analyses"
    }


# ==================== JSON-RPC 2.0 Endpoint ====================

@app.post("/rpc", response_model=JSONRPCResponse)
async def json_rpc_endpoint(rpc_request: JSONRPCRequest):
    """
    JSON-RPC 2.0 endpoint for IDE integration.
    
    Supported methods:
    - analyze(code, language, file_path, project_root)
    - getStatus()
    - submitFeedback(vulnerability_id, action, comment)
    - clearCache()
    """
    try:
        method = rpc_request.method
        params = rpc_request.params
        
        if method == "analyze":
            # Convert to AnalysisRequest
            req = AnalysisRequest(**params)
            result = await analyze_code(req, BackgroundTasks())
            return JSONRPCResponse(
                result=result.dict(),
                id=rpc_request.id
            )
        
        elif method == "getStatus":
            result = await get_status()
            return JSONRPCResponse(
                result=result,
                id=rpc_request.id
            )
        
        elif method == "submitFeedback":
            req = FeedbackRequest(**params)
            result = await submit_feedback(req)
            return JSONRPCResponse(
                result=result,
                id=rpc_request.id
            )
        
        elif method == "clearCache":
            result = await clear_cache()
            return JSONRPCResponse(
                result=result,
                id=rpc_request.id
            )
        
        else:
            return JSONRPCResponse(
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                id=rpc_request.id
            )
    
    except Exception as e:
        return JSONRPCResponse(
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=rpc_request.id
        )


# ==================== WebSocket Endpoint ====================

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    Clients receive:
    - Analysis progress
    - Vulnerability detections
    - Cache hits
    - Status updates
    """
    await websocket.accept()
    state.active_connections.append(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "StreamGuard agent connected",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0"
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
                
                # Handle JSON messages
                else:
                    try:
                        message = json.loads(data)
                        await handle_websocket_message(websocket, message)
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON"
                        })
            
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    
    finally:
        state.active_connections.remove(websocket)


async def handle_websocket_message(websocket: WebSocket, message: Dict):
    """Handle messages from WebSocket clients."""
    msg_type = message.get("type")
    
    if msg_type == "subscribe":
        # Subscribe to specific events
        await websocket.send_json({
            "type": "subscribed",
            "events": message.get("events", [])
        })
    
    elif msg_type == "get_status":
        # Send current status
        status = await get_status()
        await websocket.send_json({
            "type": "status",
            "data": status
        })


async def notify_websocket_clients(message: Dict):
    """Broadcast message to all connected WebSocket clients."""
    disconnected = []
    
    for connection in state.active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"Failed to send to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        state.active_connections.remove(conn)


# ==================== Detection Pipeline ====================

async def run_detection_pipeline(
    code: str,
    language: str,
    file_path: Optional[str],
    project_root: Optional[str],
    incremental: bool,
    previous_hash: Optional[str]
) -> List[Vulnerability]:
    """
    Run the multi-agent detection pipeline.
    
    Pipeline: Syntax → Semantic → Context → Verification
    """
    from core.engine.orchestrator import DetectionOrchestrator
    from core.explainability.explainability_engine import ExplainabilityEngine
    
    orchestrator = DetectionOrchestrator()
    
    # Run pipeline with all agents
    results = await orchestrator.analyze(
        code=code,
        language=language,
        file_path=file_path,
        project_root=project_root,
        incremental=incremental
    )
    
    # Generate explanations for each vulnerability
    explainability_engine = ExplainabilityEngine(
        model=orchestrator.semantic_agent.model,
        tokenizer=orchestrator.semantic_agent.tokenizer
    )
    
    vulnerabilities = []
    for result in results:
        # Generate explanation
        explanation = explainability_engine.explain(
            vulnerability_id=result['id'],
            code=code,
            detection_result=result,
            agent_confidences=result.get('agent_confidences', {})
        )
        
        vulnerabilities.append(Vulnerability(
            id=result['id'],
            type=result['type'],
            severity=result['severity'],
            confidence=result['confidence'],
            line=result['line'],
            column=result.get('column'),
            end_line=result.get('end_line'),
            message=result['message'],
            explanation=explanation,
            suggested_fix=result.get('suggested_fix'),
            cve_references=result.get('cve_references', [])
        ))
    
    return vulnerabilities


# ==================== Server Startup ====================

def start_agent(
    host: str = "127.0.0.1",
    port: int = 8765,
    reload: bool = False,
    workers: int = 1
):
    """Start the StreamGuard local agent server."""
    print("="*60)
    print("🚀 StreamGuard Local Agent v3.0")
    print("="*60)
    print(f"   Server: http://{host}:{port}")
    print(f"   API Docs: http://{host}:{port}/docs")
    print(f"   WebSocket: ws://{host}:{port}/stream")
    print(f"   JSON-RPC: http://{host}:{port}/rpc")
    print("="*60)
    print("   Supported Languages: Python, JavaScript, TypeScript, Java, Go")
    print("   Performance: <100ms response time")
    print("   Cache: LRU (10,000 entries)")
    print("="*60)
    
    uvicorn.run(
        "core.engine.local_agent:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
        access_log=False
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    start_agent(host=host, port=port)
```

---

### 2. Cross-Platform File System Monitor

**File:** `core/engine/file_monitor.py`

```python
"""Cross-platform file system monitoring with watchdog."""

import os
import sys
from pathlib import Path
from typing import Callable, Set, Optional, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import asyncio
import hashlib
import time

class CodeFileHandler(FileSystemEventHandler):
    """Handler for code file changes with debouncing."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.go', '.rs', '.cpp', '.c', '.h',
        '.php', '.rb', '.sql', '.cs', '.swift',
        '.kt', '.scala', '.r', '.m'
    }
    
    # Ignored directories (performance optimization)
    IGNORED_DIRS = {
        'node_modules', '.git', '__pycache__',
        'venv', 'env', 'dist', 'build',
        '.vscode', '.idea', 'target', 'out',
        '.gradle', '.npm', 'bower_components',
        'vendor', '.bundle', 'tmp', 'temp'
    }
    
    # Ignored file patterns
    IGNORED_PATTERNS = {
        '.pyc', '.pyo', '.so', '.dll', '.dylib',
        '.class', '.jar', '.war', '.min.js',
        '.map', '.lock', '.log'
    }
    
    def __init__(self, callback: Callable, debounce_delay: float = 0.5):
        super().__init__()
        self.callback = callback
        self.debounce_delay = debounce_delay
        self.pending_files: Dict[str, float] = {}  # file_path -> timestamp
        self.file_checksums: Dict[str, str] = {}
        self.loop = None
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Check if supported file type
        if not self._is_supported_file(file_path):
            return
        
        # Check if file actually changed (avoid duplicate events)
        if not self._has_file_changed(file_path):
            return
        
        # Debounce rapid changes
        current_time = time.time()
        
        if file_path in self.pending_files:
            # File already pending, update timestamp
            self.pending_files[file_path] = current_time
        else:
            # New file change
            self.pending_files[file_path] = current_time
            
            # Schedule callback
            if self.loop is None:
                self.loop = asyncio.get_event_loop()
            
            asyncio.run_coroutine_threadsafe(
                self._debounced_callback(file_path),
                self.loop
            )
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file should be monitored."""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False
        
        # Check ignored patterns
        for pattern in self.IGNORED_PATTERNS:
            if pattern in path.name:
                return False
        
        # Check if in ignored directory
        for part in path.parts:
            if part in self.IGNORED_DIRS:
                return False
        
        return True
    
    def _has_file_changed(self, file_path: str) -> bool:
        """Check if file content actually changed using checksum."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                checksum = hashlib.md5(content).hexdigest()
            
            if file_path in self.file_checksums:
                if self.file_checksums[file_path] == checksum:
                    return False  # No change
            
            self.file_checksums[file_path] = checksum
            return True
        
        except Exception as e:
            print(f"Error checking file change: {e}")
            return True  # Assume changed if error
    
    async def _debounced_callback(self, file_path: str):
        """Debounce file changes to avoid excessive analysis."""
        # Wait for debounce delay
        await asyncio.sleep(self.debounce_delay)
        
        # Check if file was modified again during debounce
        if file_path in self.pending_files:
            last_modified = self.pending_files[file_path]
            
            if time.time() - last_modified >= self.debounce_delay:
                # No modifications during debounce period, trigger callback
                self.pending_files.pop(file_path, None)
                
                try:
                    await self.callback(file_path)
                except Exception as e:
                    print(f"Error in file change callback: {e}")


class FileSystemMonitor:
    """Cross-platform file system monitor using watchdog."""
    
    def __init__(
        self,
        project_root: str,
        callback: Callable,
        debounce_delay: float = 0.5
    ):
        self.project_root = Path(project_root).resolve()
        self.callback = callback
        self.observer = Observer()
        self.handler = CodeFileHandler(callback, debounce_delay)
        self.running = False
    
    def start(self):
        """Start monitoring the project directory."""
        if self.running:
            print("⚠️  Monitor already running")
            return
        
        print(f"📁 Monitoring: {self.project_root}")
        print(f"   Supported: {', '.join(self.handler.SUPPORTED_EXTENSIONS)}")
        print(f"   Debounce: {self.handler.debounce_delay}s")
        
        self.observer.schedule(
            self.handler,
            str(self.project_root),
            recursive=True
        )
        
        self.observer.start()
        self.running = True
        print("✅ File monitor started")
    
    def stop(self):
        """Stop monitoring."""
        if not self.running:
            return
        
        print("🛑 Stopping file monitor...")
        self.observer.stop()
        self.observer.join()
        self.running = False
        print("✅ File monitor stopped")
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        return {
            "monitored_files": len(self.handler.file_checksums),
            "pending_files": len(self.handler.pending_files),
            "project_root": str(self.project_root)
        }


# ==================== Standalone Monitor ====================

async def analyze_file_callback(file_path: str):
    """
    Callback when file changes.
    
    This sends the file to the local agent for analysis.
    """
    print(f"📝 File changed: {file_path}")
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Detect language
        language = detect_language_from_extension(file_path)
        
        # Send to local agent API
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8765/analyze',
                json={
                    'code': code,
                    'language': language,
                    'file_path': file_path,
                    'incremental': True
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Analysis complete: {len(result['vulnerabilities'])} vulnerabilities")
                else:
                    print(f"❌ Analysis failed: {response.status}")
    
    except Exception as e:
        print(f"Error analyzing file: {e}")


def detect_language_from_extension(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.php': 'php',
        '.rb': 'ruby',
        '.sql': 'sql',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    return language_map.get(ext, 'unknown')


# Example standalone usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_monitor.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Create monitor
    monitor = FileSystemMonitor(
        project_root=project_root,
        callback=analyze_file_callback,
        debounce_delay=0.5
    )
    
    try:
        monitor.start()
        
        print("\n🔍 Watching for file changes... (Press Ctrl+C to stop)\n")
        
        # Keep running
        loop.run_forever()
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        monitor.stop()
    
    finally:
        loop.close()
```

---

### 3. Incremental Diff Engine

**File:** `core/engine/diff_engine.py`

```python
"""Incremental diff computation for efficient analysis."""

import difflib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript

@dataclass
class CodeDiff:
    """Represents a code diff."""
    file_path: str
    old_content: str
    new_content: str
    changed_lines: List[int]
    changed_functions: List[str]
    diff_context: str
    is_incremental: bool


@dataclass
class DiffRegion:
    """A region of changed code."""
    start_line: int
    end_line: int
    content: str
    context_before: str
    context_after: str


class IncrementalDiffEngine:
    """Compute incremental diffs for efficient analysis."""
    
    def __init__(self, context_lines: int = 20):
        self.context_lines = context_lines
        self.parsers = self._init_parsers()
        self.previous_versions: Dict[str, str] = {}
    
    def _init_parsers(self) -> Dict[str, tree_sitter.Parser]:
        """Initialize Tree-sitter parsers."""
        parsers = {}
        
        # Python parser
        python_parser = tree_sitter.Parser()
        python_parser.set_language(
            tree_sitter.Language(tree_sitter_python.language(), 'python')
        )
        parsers['python'] = python_parser
        
        # JavaScript parser
        js_parser = tree_sitter.Parser()
        js_parser.set_language(
            tree_sitter.Language(tree_sitter_javascript.language(), 'javascript')
        )
        parsers['javascript'] = js_parser
        parsers['typescript'] = js_parser  # Reuse for TypeScript
        
        return parsers
    
    def compute_diff(
        self,
        file_path: str,
        new_content: str,
        language: str
    ) -> CodeDiff:
        """
        Compute diff between previous and new version.
        
        Returns only changed regions with context.
        """
        # Get previous version
        old_content = self.previous_versions.get(file_path, "")
        
        if not old_content:
            # First time seeing this file, analyze everything
            return CodeDiff(
                file_path=file_path,
                old_content="",
                new_content=new_content,
                changed_lines=list(range(1, len(new_content.split('\n')) + 1)),
                changed_functions=[],
                diff_context=new_content,
                is_incremental=False
            )
        
        # Compute line-level diff
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        differ = difflib.Differ()
        diff = list(differ.compare(old_lines, new_lines))
        
        # Extract changed lines
        changed_lines = []
        line_num = 0
        
        for line in diff:
            if line.startswith('  '):  # Unchanged
                line_num += 1
            elif line.startswith('+ '):  # Added
                line_num += 1
                changed_lines.append(line_num)
            elif line.startswith('- '):  # Removed
                pass  # Don't increment line_num
        
        if not changed_lines:
            # No changes
            return CodeDiff(
                file_path=file_path,
                old_content=old_content,
                new_content=new_content,
                changed_lines=[],
                changed_functions=[],
                diff_context="",
                is_incremental=True
            )
        
        # Find changed functions using AST
        changed_functions = self._find_changed_functions(
            old_content,
            new_content,
            language
        )
        
        # Extract context around changes
        diff_context = self._extract_context(
            new_content,
            changed_lines,
            self.context_lines
        )
        
        # Update previous version
        self.previous_versions[file_path] = new_content
        
        return CodeDiff(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            changed_lines=changed_lines,
            changed_functions=changed_functions,
            diff_context=diff_context,
            is_incremental=True
        )
    
    def _find_changed_functions(
        self,
        old_content: str,
        new_content: str,
        language: str
    ) -> List[str]:
        """Find which functions were modified using AST diff."""
        if language not in self.parsers:
            return []
        
        parser = self.parsers[language]
        
        try:
            # Parse both versions
            old_tree = parser.parse(bytes(old_content, "utf8"))
            new_tree = parser.parse(bytes(new_content, "utf8"))
            
            # Extract function names from both
            old_functions = self._extract_function_names(old_tree.root_node, old_content)
            new_functions = self._extract_function_names(new_tree.root_node, new_content)
            
            # Find changed functions
            changed = []
            
            for func_name, func_content in new_functions.items():
                if func_name not in old_functions:
                    changed.append(func_name)  # New function
                elif old_functions[func_name] != func_content:
                    changed.append(func_name)  # Modified function
            
            return changed
        
        except Exception as e:
            print(f"Error finding changed functions: {e}")
            return []
    
    def _extract_function_names(
        self,
        node: tree_sitter.Node,
        code: str
    ) -> Dict[str, str]:
        """Extract function names and their content."""
        functions = {}
        
        def traverse(node):
            if node.type in ['function_definition', 'function_declaration', 'method_definition']:
                # Get function name
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = code[name_node.start_byte:name_node.end_byte]
                    func_content = code[node.start_byte:node.end_byte]
                    functions[func_name] = func_content
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return functions
    
    def _extract_context(
        self,
        content: str,
        changed_lines: List[int],
        context_lines: int
    ) -> str:
        """Extract code context around changed lines."""
        if not changed_lines:
            return ""
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Merge nearby changed regions
        regions = []
        current_region = None
        
        for line_num in sorted(changed_lines):
            if current_region is None:
                current_region = [line_num, line_num]
            elif line_num <= current_region[1] + context_lines * 2:
                current_region[1] = line_num
            else:
                regions.append(current_region)
                current_region = [line_num, line_num]
        
        if current_region:
            regions.append(current_region)
        
        # Extract context for each region
        context_parts = []
        
        for start, end in regions:
            # Add context before and after
            context_start = max(0, start - context_lines - 1)
            context_end = min(total_lines, end + context_lines)
            
            region_lines = lines[context_start:context_end]
            context_parts.append('\n'.join(region_lines))
        
        return '\n...\n'.join(context_parts)
    
    def get_diff_regions(self, diff: CodeDiff) -> List[DiffRegion]:
        """Get structured diff regions with context."""
        if not diff.changed_lines:
            return []
        
        lines = diff.new_content.split('\n')
        regions = []
        
        # Group nearby changes
        current_start = diff.changed_lines[0]
        current_end = diff.changed_lines[0]
        
        for line_num in diff.changed_lines[1:]:
            if line_num <= current_end + self.context_lines * 2:
                current_end = line_num
            else:
                # Create region
                regions.append(self._create_region(lines, current_start, current_end))
                current_start = line_num
                current_end = line_num
        
        # Add last region
        if current_start:
            regions.append(self._create_region(lines, current_start, current_end))
        
        return regions
    
    def _create_region(
        self,
        lines: List[str],
        start_line: int,
        end_line: int
    ) -> DiffRegion:
        """Create a DiffRegion with context."""
        total_lines = len(lines)
        
        # Get context
        context_start = max(0, start_line - self.context_lines - 1)
        context_end = min(total_lines, end_line + self.context_lines)
        
        # Extract content
        content = '\n'.join(lines[start_line-1:end_line])
        context_before = '\n'.join(lines[context_start:start_line-1])
        context_after = '\n'.join(lines[end_line:context_end])
        
        return DiffRegion(
            start_line=start_line,
            end_line=end_line,
            content=content,
            context_before=context_before,
            context_after=context_after
        )
    
    def clear_cache(self):
        """Clear cached previous versions."""
        self.previous_versions.clear()


# Example usage
if __name__ == "__main__":
    engine = IncrementalDiffEngine(context_lines=20)
    
    # Simulate file changes
    old_code = """
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "'"
    result = execute_query(query)
    return result
"""
    
    new_code = """
def login(username, password):
    query = "SELECT * FROM users WHERE username=?"
    result = execute_query(query, (username,))
    return result
"""
    
    # Compute diff
    diff = engine.compute_diff(
        file_path="auth.py",
        new_content=new_code,
        language="python"
    )
    
    print(f"Changed lines: {diff.changed_lines}")
    print(f"Changed functions: {diff.changed_functions}")
    print(f"Is incremental: {diff.is_incremental}")
    print(f"\nDiff context:\n{diff.diff_context}")
```

---

### 4. IDE Integration Examples

#### VS Code Extension

**File:** `ide-plugins/vscode/extension.js`

```javascript
// VS Code Extension for StreamGuard
const vscode = require('vscode');
const axios = require('axios');

const AGENT_URL = 'http://localhost:8765';

let diagnosticCollection;
let ws = null;

function activate(context) {
    console.log('StreamGuard extension activated');
    
    // Create diagnostic collection
    diagnosticCollection = vscode.languages.createDiagnosticCollection('streamguard');
    context.subscriptions.push(diagnosticCollection);
    
    // Connect to WebSocket
    connectWebSocket();
    
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('streamguard.analyzeFile', analyzeCurrentFile)
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('streamguard.clearCache', clearCache)
    );
    
    // Analyze on save
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(document => {
            analyzeDocument(document);
        })
    );
    
    // Analyze on open
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(document => {
            analyzeDocument(document);
        })
    );
}

async function analyzeDocument(document) {
    const languageId = document.languageId;
    
    // Check if supported language
    const supported = ['python', 'javascript', 'typescript', 'java', 'go'];
    if (!supported.includes(languageId)) {
        return;
    }
    
    const code = document.getText();
    const filePath = document.fileName;
    
    try {
        // Send to agent
        const response = await axios.post(`${AGENT_URL}/analyze`, {
            code: code,
            language: languageId,
            file_path: filePath,
            incremental: true
        });
        
        // Update diagnostics
        updateDiagnostics(document, response.data.vulnerabilities);
        
        // Show status
        vscode.window.setStatusBarMessage(
            `StreamGuard: ${response.data.vulnerabilities.length} vulnerabilities found`,
            3000
        );
    } catch (error) {
        vscode.window.showErrorMessage(`StreamGuard: ${error.message}`);
    }
}

function updateDiagnostics(document, vulnerabilities) {
    const diagnostics = vulnerabilities.map(vuln => {
        const range = new vscode.Range(
            vuln.line - 1, vuln.column || 0,
            vuln.end_line ? vuln.end_line - 1 : vuln.line - 1, 100
        );
        
        const diagnostic = new vscode.Diagnostic(
            range,
            vuln.message,
            severityMap(vuln.severity)
        );
        
        diagnostic.source = 'StreamGuard';
        diagnostic.code = vuln.type;
        
        return diagnostic;
    });
    
    diagnosticCollection.set(document.uri, diagnostics);
}

function severityMap(severity) {
    switch (severity) {
        case 'critical':
        case 'high':
            return vscode.DiagnosticSeverity.Error;
        case 'medium':
            return vscode.DiagnosticSeverity.Warning;
        case 'low':
            return vscode.DiagnosticSeverity.Information;
        default:
            return vscode.DiagnosticSeverity.Hint;
    }
}

function connectWebSocket() {
    const WebSocket = require('ws');
    
    ws = new WebSocket(`ws://localhost:8765/stream`);
    
    ws.on('open', () => {
        console.log('StreamGuard WebSocket connected');
    });
    
    ws.on('message', (data) => {
        const message = JSON.parse(data);
        
        if (message.type === 'analysis_complete') {
            vscode.window.showInformationMessage(
                `StreamGuard: Found ${message.vulnerabilities_count} vulnerabilities`
            );
        }
    });
    
    ws.on('close', () => {
        console.log('StreamGuard WebSocket disconnected');
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    });
}

async function clearCache() {
    try {
        await axios.delete(`${AGENT_URL}/cache`);
        vscode.window.showInformationMessage('StreamGuard cache cleared');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to clear cache: ${error.message}`);
    }
}

function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        analyzeDocument(editor.document);
    }
}

function deactivate() {
    if (ws) {
        ws.close();
    }
}

module.exports = {
    activate,
    deactivate
};
```

#### Generic IDE Integration (JSON-RPC)

**File:** `ide-plugins/generic/streamguard_client.py`

```python
"""Generic StreamGuard client using JSON-RPC.

Works with any IDE that can execute Python scripts.
"""

import json
import requests
from typing import Dict, List, Optional

class StreamGuardClient:
    """Client for StreamGuard local agent using JSON-RPC 2.0."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.url = f"http://{host}:{port}/rpc"
        self.request_id = 0
    
    def _call(self, method: str, params: Dict) -> Dict:
        """Make JSON-RPC 2.0 call."""
        self.request_id += 1
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id
        }
        
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            raise Exception(f"RPC Error: {result['error']}")
        
        return result["result"]
    
    def analyze(
        self,
        code: str,
        language: str,
        file_path: Optional[str] = None,
        project_root: Optional[str] = None
    ) -> Dict:
        """Analyze code for vulnerabilities."""
        return self._call("analyze", {
            "code": code,
            "language": language,
            "file_path": file_path,
            "project_root": project_root,
            "incremental": True
        })
    
    def get_status(self) -> Dict:
        """Get agent status."""
        return self._call("getStatus", {})
    
    def submit_feedback(
        self,
        vulnerability_id: str,
        action: str,
        comment: Optional[str] = None
    ) -> Dict:
        """Submit feedback on a vulnerability."""
        return self._call("submitFeedback", {
            "vulnerability_id": vulnerability_id,
            "action": action,
            "comment": comment
        })
    
    def clear_cache(self) -> Dict:
        """Clear analysis cache."""
        return self._call("clearCache", {})


# Example usage
if __name__ == "__main__":
    client = StreamGuardClient()
    
    # Analyze some code
    code = """
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "'"
    execute(query)
"""
    
    result = client.analyze(
        code=code,
        language="python",
        file_path="test.py"
    )
    
    print(f"Found {len(result['vulnerabilities'])} vulnerabilities")
    
    for vuln in result['vulnerabilities']:
        print(f"  Line {vuln['line']}: {vuln['message']}")
```

---

### 5. Docker Deployment

#### Dockerfile

**File:** `Dockerfile`

```dockerfile
# StreamGuard Local Agent Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ ./core/
COPY training/ ./training/
COPY models/ ./models/

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run agent
CMD ["python", "-m", "core.engine.local_agent"]
```

#### Docker Compose

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  streamguard-agent:
    build: .
    container_name: streamguard-agent
    ports:
      - "8765:8765"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=info
    volumes:
      # Mount project directory for analysis
      - ./:/workspace:ro
      # Mount models directory
      - ./models:/app/models:ro
    networks:
      - streamguard-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  neo4j:
    image: neo4j:5.14-community
    container_name: streamguard-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/streamguard
      - NEO4J_PLUGINS=["graph-data-science", "apoc"]
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    networks:
      - streamguard-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: streamguard-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - streamguard-network
    restart: unless-stopped
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

volumes:
  neo4j_data:
  neo4j_logs:
  redis_data:

networks:
  streamguard-network:
    driver: bridge
```

#### Docker Launch Script

**File:** `scripts/docker_start.sh`

```bash
#!/bin/bash
# Launch StreamGuard with Docker

set -e

echo "🚀 Starting StreamGuard Local Agent with Docker"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build image
echo "📦 Building Docker image..."
docker-compose build

# Start services
echo "🔄 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check health
echo "🔍 Checking service health..."
docker-compose ps

# Test API
echo ""
echo "🧪 Testing API..."
curl -s http://localhost:8765/health | python -m json.tool

echo ""
echo "✅ StreamGuard is running!"
echo ""
echo "Services:"
echo "  • Agent API:    http://localhost:8765"
echo "  • API Docs:     http://localhost:8765/docs"
echo "  • WebSocket:    ws://localhost:8765/stream"
echo "  • Neo4j:        http://localhost:7474"
echo "  • Redis:        localhost:6379"
echo ""
echo "Commands:"
echo "  • View logs:    docker-compose logs -f streamguard-agent"
echo "  • Stop:         docker-compose down"
echo "  • Restart:      docker-compose restart"
```

---

### 6. Testing & Benchmarks

#### Integration Tests

**File:** `tests/integration/test_local_agent.py`

```python
"""Integration tests for local agent."""

import pytest
import asyncio
import aiohttp
import websockets
from pathlib import Path
import time

BASE_URL = "http://localhost:8765"
WS_URL = "ws://localhost:8765/stream"


class TestLocalAgent:
    """Test local agent API."""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_analyze_python_code(self):
        """Test analyzing Python code."""
        code = '''
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "'"
    execute(query)
'''
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/analyze",
                json={
                    'code': code,
                    'language': 'python',
                    'file_path': 'test.py',
                    'incremental': False
                }
            ) as response:
                assert response.status == 200
                data = await response.json()
                
                assert data['status'] == 'success'
                assert 'vulnerabilities' in data
                assert len(data['vulnerabilities']) > 0
                assert data['analysis_time_ms'] < 1000  # <1s
                
                # Check vulnerability structure
                vuln = data['vulnerabilities'][0]
                assert 'id' in vuln
                assert 'type' in vuln
                assert 'severity' in vuln
                assert 'confidence' in vuln
                assert 'line' in vuln
                assert 'message' in vuln
    
    @pytest.mark.asyncio
    async def test_incremental_analysis_cache(self):
        """Test incremental analysis with caching."""
        code = "def test(): pass"
        
        async with aiohttp.ClientSession() as session:
            # First analysis
            start = time.time()
            async with session.post(
                f"{BASE_URL}/analyze",
                json={
                    'code': code,
                    'language': 'python',
                    'incremental': True
                }
            ) as response:
                data1 = await response.json()
                time1 = time.time() - start
            
            # Second analysis (should be cached)
            start = time.time()
            async with session.post(
                f"{BASE_URL}/analyze",
                json={
                    'code': code,
                    'language': 'python',
                    'incremental': True
                }
            ) as response:
                data2 = await response.json()
                time2 = time.time() - start
            
            # Cache should be faster
            assert data2['cached'] == True
            assert time2 < time1
            assert data1['file_hash'] == data2['file_hash']
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        async with websockets.connect(WS_URL) as websocket:
            # Receive welcome message
            message = await websocket.recv()
            data = eval(message)  # Note: use json.loads in production
            
            assert data['type'] == 'connected'
            assert 'version' in data
            
            # Send ping
            await websocket.send('ping')
            response = await websocket.recv()
            assert response == 'pong'
    
    @pytest.mark.asyncio
    async def test_json_rpc_endpoint(self):
        """Test JSON-RPC endpoint."""
        async with aiohttp.ClientSession() as session:
            # Call getStatus method
            async with session.post(
                f"{BASE_URL}/rpc",
                json={
                    'jsonrpc': '2.0',
                    'method': 'getStatus',
                    'params': {},
                    'id': 1
                }
            ) as response:
                assert response.status == 200
                data = await response.json()
                
                assert data['jsonrpc'] == '2.0'
                assert 'result' in data
                assert data['id'] == 1
    
    @pytest.mark.asyncio
    async def test_submit_feedback(self):
        """Test feedback submission."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/feedback",
                json={
                    'vulnerability_id': 'test_vuln_001',
                    'action': 'accepted',
                    'comment': 'Good catch!'
                }
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert data['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{BASE_URL}/cache") as response:
                assert response.status == 200
                data = await response.json()
                assert data['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting."""
        async with aiohttp.ClientSession() as session:
            # Send many requests rapidly
            tasks = []
            for _ in range(150):  # Exceed 100 req/min limit
                task = session.post(
                    f"{BASE_URL}/analyze",
                    json={'code': 'test', 'language': 'python'}
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have some rate limit errors
            rate_limited = sum(
                1 for r in responses 
                if not isinstance(r, Exception) and r.status == 429
            )
            assert rate_limited > 0


class TestFileMonitor:
    """Test file system monitor."""
    
    @pytest.mark.asyncio
    async def test_file_change_detection(self, tmp_path):
        """Test file change detection."""
        from core.engine.file_monitor import FileSystemMonitor
        
        changes_detected = []
        
        async def callback(file_path: str):
            changes_detected.append(file_path)
        
        # Create monitor
        monitor = FileSystemMonitor(
            project_root=str(tmp_path),
            callback=callback,
            debounce_delay=0.1
        )
        
        monitor.start()
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        # Wait for detection
        await asyncio.sleep(0.5)
        
        # Modify file
        test_file.write_text("def test(): return True")
        
        # Wait for detection
        await asyncio.sleep(0.5)
        
        monitor.stop()
        
        # Should have detected changes
        assert len(changes_detected) > 0
        assert str(test_file) in changes_detected


class TestDiffEngine:
    """Test incremental diff engine."""
    
    def test_compute_diff(self):
        """Test diff computation."""
        from core.engine.diff_engine import IncrementalDiffEngine
        
        engine = IncrementalDiffEngine()
        
        old_code = "def test():\n    pass"
        new_code = "def test():\n    return True"
        
        # First time - full analysis
        diff1 = engine.compute_diff("test.py", old_code, "python")
        assert diff1.is_incremental == False
        
        # Second time - incremental
        diff2 = engine.compute_diff("test.py", new_code, "python")
        assert diff2.is_incremental == True
        assert len(diff2.changed_lines) > 0
    
    def test_function_change_detection(self):
        """Test detecting changed functions."""
        from core.engine.diff_engine import IncrementalDiffEngine
        
        engine = IncrementalDiffEngine()
        
        old_code = """
def login(user):
    query = "SELECT * FROM users WHERE id=" + user
    return execute(query)

def logout(user):
    return True
"""
        
        new_code = """
def login(user):
    query = "SELECT * FROM users WHERE id=?"
    return execute(query, (user,))

def logout(user):
    return True
"""
        
        # Compute diff
        engine.compute_diff("test.py", old_code, "python")
        diff = engine.compute_diff("test.py", new_code, "python")
        
        # Should detect login function changed, but not logout
        assert 'login' in diff.changed_functions
        assert 'logout' not in diff.changed_functions


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
```

#### Performance Benchmarks

**File:** `tests/benchmarks/benchmark_agent.py`

```python
"""Performance benchmarks for local agent."""

import asyncio
import aiohttp
import time
from statistics import mean, median, stdev
from typing import List

BASE_URL = "http://localhost:8765"


async def benchmark_analysis_latency(num_requests: int = 100):
    """Benchmark analysis latency."""
    print(f"🏃 Benchmarking analysis latency ({num_requests} requests)...")
    
    code = """
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "'"
    result = execute(query)
    return result
"""
    
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            start = time.time()
            
            async with session.post(
                f"{BASE_URL}/analyze",
                json={
                    'code': code,
                    'language': 'python',
                    'incremental': False
                }
            ) as response:
                await response.json()
            
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_requests}")
    
    # Calculate statistics
    print("\n📊 Latency Statistics:")
    print(f"  Mean:     {mean(latencies):.2f} ms")
    print(f"  Median:   {median(latencies):.2f} ms")
    print(f"  Min:      {min(latencies):.2f} ms")
    print(f"  Max:      {max(latencies):.2f} ms")
    print(f"  Std Dev:  {stdev(latencies):.2f} ms")
    print(f"  P95:      {sorted(latencies)[int(len(latencies) * 0.95)]:.2f} ms")
    print(f"  P99:      {sorted(latencies)[int(len(latencies) * 0.99)]:.2f} ms")
    
    # Check if meets target (<100ms average)
    if mean(latencies) < 100:
        print("\n✅ Performance target met (<100ms average)")
    else:
        print("\n⚠️  Performance target not met (>100ms average)")
    
    return latencies


async def benchmark_cache_performance():
    """Benchmark cache hit performance."""
    print("\n🏃 Benchmarking cache performance...")
    
    code = "def test(): pass"
    
    async with aiohttp.ClientSession() as session:
        # First request (miss)
        start = time.time()
        async with session.post(
            f"{BASE_URL}/analyze",
            json={'code': code, 'language': 'python', 'incremental': True}
        ) as response:
            data1 = await response.json()
        miss_time = (time.time() - start) * 1000
        
        # Second request (hit)
        start = time.time()
        async with session.post(
            f"{BASE_URL}/analyze",
            json={'code': code, 'language': 'python', 'incremental': True}
        ) as response:
            data2 = await response.json()
        hit_time = (time.time() - start) * 1000
    
    speedup = miss_time / hit_time
    
    print(f"\n📊 Cache Performance:")
    print(f"  Cache miss:  {miss_time:.2f} ms")
    print(f"  Cache hit:   {hit_time:.2f} ms")
    print(f"  Speedup:     {speedup:.1f}x")
    
    if hit_time < 10:
        print("\n✅ Cache performance excellent (<10ms)")
    else:
        print("\n⚠️  Cache performance could be improved")


async def benchmark_throughput(duration_seconds: int = 10):
    """Benchmark request throughput."""
    print(f"\n🏃 Benchmarking throughput ({duration_seconds}s)...")
    
    code = "def test(): pass"
    request_count = 0
    start_time = time.time()
    
    async def make_request(session):
        nonlocal request_count
        async with session.post(
            f"{BASE_URL}/analyze",
            json={'code': code, 'language': 'python', 'incremental': True}
        ) as response:
            await response.json()
            request_count += 1
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        while time.time() - start_time < duration_seconds:
            # Send 10 concurrent requests
            for _ in range(10):
                tasks.append(make_request(session))
            
            await asyncio.gather(*tasks)
            tasks.clear()
    
    elapsed = time.time() - start_time
    throughput = request_count / elapsed
    
    print(f"\n📊 Throughput:")
    print(f"  Total requests:  {request_count}")
    print(f"  Duration:        {elapsed:.2f}s")
    print(f"  Throughput:      {throughput:.2f} req/s")
    
    if throughput > 50:
        print("\n✅ Throughput excellent (>50 req/s)")
    elif throughput > 20:
        print("\n✅ Throughput good (>20 req/s)")
    else:
        print("\n⚠️  Throughput could be improved")


async def run_all_benchmarks():
    """Run all benchmarks."""
    print("="*60)
    print("StreamGuard Local Agent Performance Benchmarks")
    print("="*60)
    
    # 1. Latency
    await benchmark_analysis_latency(num_requests=100)
    
    # 2. Cache
    await benchmark_cache_performance()
    
    # 3. Throughput
    await benchmark_throughput(duration_seconds=10)
    
    print("\n" + "="*60)
    print("Benchmarks complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
```

---

## ✅ Implementation Checklist

### Core Components
- [ ] FastAPI server with REST endpoints
- [ ] JSON-RPC 2.0 support
- [ ] WebSocket real-time streaming
- [ ] CORS and rate limiting middleware
- [ ] Request validation with Pydantic
- [ ] LRU cache implementation
- [ ] Error handling and logging

### File System Monitoring
- [ ] Cross-platform file monitor (watchdog)
- [ ] File type filtering
- [ ] Directory ignore patterns
- [ ] Debouncing logic
- [ ] Checksum-based change detection
- [ ] Async event handling

### Incremental Analysis
- [ ] Diff computation engine
- [ ] AST-based function change detection
- [ ] Context extraction
- [ ] Previous version caching
- [ ] Region-based analysis

### Integration
- [ ] VS Code extension example
- [ ] Generic JSON-RPC client
- [ ] CLI tool
- [ ] Docker deployment
- [ ] Docker Compose setup

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests (API, WebSocket, etc.)
- [ ] Performance benchmarks
- [ ] Load testing
- [ ] End-to-end scenarios

### Documentation
- [ ] API documentation (OpenAPI/Swagger)
- [ ] WebSocket protocol documentation
- [ ] JSON-RPC method reference
- [ ] IDE integration guides
- [ ] Deployment guides

---

## 🎯 Performance Targets

| Metric | Target | Maximum | Status |
|--------|--------|---------|--------|
| **Analysis Latency (avg)** | <50ms | <100ms | ⏳ To Validate |
| **Analysis Latency (P95)** | <200ms | <500ms | ⏳ To Validate |
| **Cache Hit Latency** | <5ms | <10ms | ⏳ To Validate |
| **WebSocket Latency** | <10ms | <50ms | ⏳ To Validate |
| **Throughput** | >50 req/s | >20 req/s | ⏳ To Validate |
| **Memory Usage** | <500MB | <2GB | ⏳ To Validate |
| **File Monitor Overhead** | <1% CPU | <5% CPU | ⏳ To Validate |
| **Cache Hit Rate** | >80% | >60% | ⏳ To Validate |

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start agent
python -m core.engine.local_agent

# 3. In another terminal, start file monitor
python -m core.engine.file_monitor /path/to/project

# 4. Test API
curl http://localhost:8765/health

# 5. Analyze code
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def test(): pass",
    "language": "python"
  }'
```

### Docker Deployment

```bash
# 1. Start with Docker Compose
./scripts/docker_start.sh

# 2. View logs
docker-compose logs -f streamguard-agent

# 3. Stop
docker-compose down
```

### IDE Integration

**VS Code:**
```bash
# Install extension
cd ide-plugins/vscode
npm install
npm run compile
code --install-extension .
```

**Any IDE (JSON-RPC):**
```python
from ide_plugins.generic.streamguard_client import StreamGuardClient

client = StreamGuardClient()
result = client.analyze(code="...", language="python")
print(result['vulnerabilities'])
```

---

## 🔧 Troubleshooting

### Agent won't start

```bash
# Check if port is in use
lsof -i :8765

# Kill existing process
kill -9 $(lsof -t -i :8765)

# Start with different port
python -m core.engine.local_agent 127.0.0.1 8766
```

### WebSocket connection fails

```bash
# Test WebSocket
pip install websockets
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8765/stream') as ws:
        msg = await ws.recv()
        print(msg)

asyncio.run(test())
"
```

### Slow analysis performance

```bash
# Check cache hit rate
curl http://localhost:8765/status | jq '.cache.hit_rate'

# Clear cache if needed
curl -X DELETE http://localhost:8765/cache

# Profile performance
python tests/benchmarks/benchmark_agent.py
```

---

## 📚 Next Steps

**After completing this phase:**

1. **Test Agent:**
   ```bash
   # Run tests
   pytest tests/integration/test_local_agent.py -v
   
   # Run benchmarks
   python tests/benchmarks/benchmark_agent.py
   ```

2. **Deploy:**
   ```bash
   # Docker deployment
   ./scripts/docker_start.sh
   ```

3. **Continue to Phase 4:**
   - [05_repository_graph.md](./05_repository_graph.md) - Repository graph system with Neo4j
   - Focus on dependency tracking and vulnerability propagation

---

## 💡 Tips for Claude Code

```bash
# Implement agent with Claude
claude --agent backend "Implement FastAPI local agent with all endpoints"

# Add WebSocket support
claude "Add WebSocket streaming to local agent with heartbeat"

# Optimize performance
claude "Profile agent performance and optimize to <100ms response time"

# Create Docker setup
claude "Create production-ready Docker deployment with compose"

# Build IDE integration
claude --agent frontend "Create VS Code extension for StreamGuard agent"

# Test everything
claude --agent testing "Create comprehensive integration tests for local agent"
```

---

**Status:** ✅ Ready for Implementation  
**Next:** [05_repository_graph.md](./05_repository_graph.md) - Repository Graph System