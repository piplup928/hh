"""Document persistence (TipTap/ProseMirror JSON + paper & typography settings)."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..config import DOCS_DIR
from ..models.schemas import DocumentModel

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _path(doc_id: str):
    return DOCS_DIR / f"{doc_id}.json"


@router.post("")
async def save_document(doc: DocumentModel):
    doc_id = doc.docId or uuid.uuid4().hex[:12]
    data = doc.model_dump()
    data["docId"] = doc_id
    data["updatedAt"] = time.time()
    _path(doc_id).write_text(json.dumps(data))
    return {"docId": doc_id, "updatedAt": data["updatedAt"]}


@router.get("")
async def list_documents():
    out = []
    for p in sorted(DOCS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            out.append({"docId": d.get("docId", p.stem), "title": d.get("title"),
                        "updatedAt": d.get("updatedAt"), "styleId": d.get("styleId")})
        except (json.JSONDecodeError, OSError):
            continue
    return out


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    p = _path(doc_id)
    if not p.exists():
        raise HTTPException(404, "document not found")
    return json.loads(p.read_text())


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    p = _path(doc_id)
    if not p.exists():
        raise HTTPException(404, "document not found")
    p.unlink()
    return {"deleted": doc_id}
