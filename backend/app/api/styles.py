"""Style endpoints: upload handwriting samples -> learn a personalized style."""
from __future__ import annotations

import io
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from ..engine.style.profile import store

router = APIRouter(prefix="/api/styles", tags=["styles"])

MAX_UPLOAD_MB = 25


@router.post("")
async def create_style(files: List[UploadFile] = File(...),
                       name: str = Form("My handwriting")):
    if not files:
        raise HTTPException(400, "upload at least one handwriting sample image")
    images = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB}MB")
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(415, f"{f.filename} is not a readable image")
        images.append(img)
    profile = store.create(images, name=name)
    return profile.to_dict()


@router.get("")
async def list_styles():
    return [p.to_dict() for p in store.list()]


@router.get("/{style_id}")
async def get_style(style_id: str):
    p = store.get(style_id)
    if p is None:
        raise HTTPException(404, "style not found")
    return p.to_dict()


@router.get("/{style_id}/canvas/{index}")
async def get_style_canvas(style_id: str, index: int):
    p = store.get(style_id)
    if p is None:
        raise HTTPException(404, "style not found")
    paths = p.canvas_paths
    if not (0 <= index < len(paths)):
        raise HTTPException(404, "canvas index out of range")
    return FileResponse(paths[index], media_type="image/png")


@router.delete("/{style_id}")
async def delete_style(style_id: str):
    if not store.delete(style_id):
        raise HTTPException(404, "style not found")
    return {"deleted": style_id}
