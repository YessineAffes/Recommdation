from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chroma_db_taxonomy"
DOCS_PATH = ROOT / "rag" / "documents"

COLLECTIONS = ["types_verres", "indices", "traitements", "couleurs"]


def _client() -> chromadb.PersistentClient:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))


def _classify_doc(filename: str) -> str | None:
    lower = filename.lower()
    if "type" in lower and "verre" in lower:
        return "types_verres"
    if "indice" in lower:
        return "indices"
    if "traitement" in lower or "crizal" in lower:
        return "traitements"
    if "couleur" in lower or "transition" in lower or "solaire" in lower or "polaris" in lower:
        return "couleurs"
    return None


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def ingest_folder(folder: Path | None = None, reset: bool = False) -> None:
    folder = folder or DOCS_PATH
    folder.mkdir(parents=True, exist_ok=True)

    client = _client()
    if reset:
        for name in COLLECTIONS:
            try:
                client.delete_collection(name)
            except Exception:
                pass

    for name in COLLECTIONS:
        client.get_or_create_collection(name)

    for docx_path in folder.glob("*.docx"):
        category = _classify_doc(docx_path.name)
        if category is None:
            continue

        d = Document(str(docx_path))
        lines: list[str] = []
        for p in d.paragraphs:
            t = p.text.strip()
            if t:
                lines.append(t)
        text = "\n".join(lines)
        chunks = _split_lines(text)

        coll = client.get_collection(category)
        existing = set(coll.get().get("ids", []))
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, 1):
            cid = f"{docx_path.stem}_{idx}"
            if cid in existing:
                continue
            ids.append(cid)
            docs.append(chunk)
            metas.append({"collection": category, "source": docx_path.name})

        if ids:
            coll.add(ids=ids, documents=docs, metadatas=metas)


def get_justification(recommendation: dict[str, Any], n: int = 2) -> list[str]:
    """Recupere des extraits par taxonomie, filtres par collection selon les besoins."""
    client = _client()
    out: list[str] = []

    mapping = [
        ("types_verres", recommendation.get("type_verre", "")),
        ("indices", recommendation.get("indice", "")),
        ("traitements", recommendation.get("traitement", "")),
        ("couleurs", recommendation.get("couleur", "")),
    ]

    for coll_name, query in mapping:
        if not query:
            continue
        coll = client.get_or_create_collection(coll_name)
        if coll.count() == 0:
            continue
        res = coll.query(
            query_texts=[str(query)],
            n_results=n,
            where={"collection": coll_name},
        )
        out.extend(res.get("documents", [[]])[0])

    return out
