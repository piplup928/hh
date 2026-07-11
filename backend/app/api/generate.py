"""Generation endpoints.

REST for one-shot generation/export, WebSocket for the live typing pipeline:

  client -> {"op":"gen","reqId":..,"text":..,"styleId":..,"marks":{..},
             "quality":"live","seed":123}
  server -> {"op":"ink","reqId":..,"results":[InkResult payload, ...]}
  server -> {"op":"error","reqId":..,"detail":...}

The client debounces typing (word boundaries); the server cancels nothing —
caching makes repeated requests cheap, and per-request ids let the client
drop stale responses.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..engine.base import EngineUnavailable
from ..engine.router import Run
from ..engine.service import service
from ..models.schemas import GenerateRequest, MathGenerateRequest

log = logging.getLogger("api.generate")
router = APIRouter(prefix="/api/generate", tags=["generate"])


@router.get("/status")
async def engine_status():
    return service.status()


@router.post("/text")
async def generate_text(req: GenerateRequest):
    try:
        results = await service.generate_text(
            req.text, req.styleId, seed=req.seed, quality=req.quality,
            marks=req.marks.model_dump() if req.marks else None)
    except KeyError:
        raise HTTPException(404, f"style {req.styleId} not found")
    except EngineUnavailable as e:
        raise HTTPException(503, str(e))
    return {"results": [r.to_payload() for r in results]}


@router.post("/math")
async def generate_math(req: MathGenerateRequest):
    try:
        result = await service.generate_run(
            Run("math", req.latex), req.styleId, seed=req.seed,
            quality=req.quality,
            marks=req.marks.model_dump() if req.marks else None)
    except KeyError:
        raise HTTPException(404, f"style {req.styleId} not found")
    except EngineUnavailable as e:
        raise HTTPException(503, str(e))
    return result.to_payload()


@router.websocket("/ws")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    tasks: set[asyncio.Task] = set()

    async def handle(msg: dict):
        req_id = msg.get("reqId")
        try:
            results = await service.generate_text(
                msg["text"], msg["styleId"], seed=msg.get("seed"),
                quality=msg.get("quality", "live"), marks=msg.get("marks"))
            await ws.send_text(json.dumps({
                "op": "ink", "reqId": req_id,
                "results": [r.to_payload() for r in results]}))
        except EngineUnavailable as e:
            await ws.send_text(json.dumps(
                {"op": "error", "reqId": req_id, "code": 503, "detail": str(e)}))
        except KeyError as e:
            await ws.send_text(json.dumps(
                {"op": "error", "reqId": req_id, "code": 404, "detail": str(e)}))
        except Exception as e:  # noqa: BLE001
            log.exception("ws generation failed")
            await ws.send_text(json.dumps(
                {"op": "error", "reqId": req_id, "code": 500, "detail": str(e)}))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"op": "error", "detail": "bad json"}))
                continue
            if msg.get("op") == "ping":
                await ws.send_text(json.dumps({"op": "pong"}))
                continue
            if msg.get("op") == "status":
                await ws.send_text(json.dumps(
                    {"op": "status", "status": service.status()}))
                continue
            if msg.get("op") == "gen" and msg.get("text") is not None \
                    and msg.get("styleId"):
                t = asyncio.create_task(handle(msg))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
