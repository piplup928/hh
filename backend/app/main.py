"""Handwriting Intelligence Platform — FastAPI backend.

Pipeline: upload handwriting -> style extraction -> personalized profile ->
live editor typing -> Paragraph-LDM (prose) + DiffInk (math) -> styled ink ->
real benchmarks (HWD / style-mAP / CER) against the uploaded samples.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import benchmarks, documents, generate, styles
from .config import ENGINE_MODE
from .engine.service import service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Handwriting Intelligence Platform",
    description="AI text-to-handwriting: Paragraph-LDM + DiffInk generation, "
                "zero-shot style imitation, real HWD/mAP benchmarks.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(styles.router)
app.include_router(generate.router)
app.include_router(documents.router)
app.include_router(benchmarks.router)


@app.get("/api/health")
async def health():
    return {"ok": True, "mode": ENGINE_MODE, "engines": service.status()}
