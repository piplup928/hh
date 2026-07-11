"""Pydantic API schemas."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MarksModel(BaseModel):
    bold: bool = False
    italic: bool = False
    color: Optional[str] = None      # "#rrggbb"
    sizeFactor: float = 1.0


class GenerateRequest(BaseModel):
    text: str = Field(..., max_length=20000)
    styleId: str
    seed: Optional[int] = None
    quality: Literal["live", "final"] = "live"
    marks: Optional[MarksModel] = None


class MathGenerateRequest(BaseModel):
    latex: str = Field(..., max_length=2000)
    styleId: str
    seed: Optional[int] = None
    quality: Literal["live", "final"] = "live"
    marks: Optional[MarksModel] = None


class DocumentModel(BaseModel):
    docId: Optional[str] = None
    title: str = "Untitled"
    content: dict = Field(default_factory=dict)   # TipTap/ProseMirror JSON
    paper: dict = Field(default_factory=dict)     # paper simulation settings
    typography: dict = Field(default_factory=dict)
    styleId: Optional[str] = None


class BenchmarkRequest(BaseModel):
    styleId: str
    # texts rendered for evaluation; defaults chosen to cover prose + math
    texts: Optional[List[str]] = None
    seed: int = 42
    maxSamples: int = 16
    metrics: List[Literal["hwd", "style_map", "fid", "bfid", "kid", "cer"]] = [
        "hwd", "style_map", "cer"]
