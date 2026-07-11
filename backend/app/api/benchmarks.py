"""Benchmark endpoints — run and inspect real style-fidelity evaluations."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from ..benchmarks.runner import BenchmarkIntegrityError, runner
from ..engine.base import EngineUnavailable
from ..models.schemas import BenchmarkRequest

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.post("/run")
async def run_benchmark(req: BenchmarkRequest):
    loop = asyncio.get_running_loop()
    try:
        report = await loop.run_in_executor(None, lambda: runner.run(
            req.styleId, texts=req.texts, seed=req.seed,
            max_samples=req.maxSamples, metrics=list(req.metrics)))
    except KeyError:
        raise HTTPException(404, f"style {req.styleId} not found")
    except BenchmarkIntegrityError as e:
        raise HTTPException(409, str(e))
    except EngineUnavailable as e:
        raise HTTPException(503, str(e))
    except ImportError as e:
        raise HTTPException(
            503, f"hwd package not installed ({e}); run: pip install -e backend/vendor/HWD")
    return report


@router.get("/reports")
async def list_reports():
    return runner.list_reports()
