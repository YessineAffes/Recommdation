from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chroma_db_taxonomy"
SCENARIO_DB_PATH = ROOT / "chroma_db"
DOCS_PATH = ROOT / "rag" / "documents"

COLLECTIONS = ["types_verres", "indices", "traitements", "couleurs"]

TYPE_KEYWORDS = [
    "simple foyer",
    "eyezen",
    "varilux",
    "liberty",
    "comfort",
    "physio",
    "x design",
    "s design",
    "xr",
    "digitime",
]
INDEX_KEYWORDS = ["indice", "1.50", "1.56", "1.60", "1.67", "1.74"]
TREATMENT_KEYWORDS = ["traitement", "crizal", "sapphire", "prevencia", "drive", "rock", "easy pro"]
COLOR_KEYWORDS = [
    "couleur",
    "blanc",
    "transition",
    "xtractive",
    "solaire",
    "polaris",
    "polarisant",
    "photochromique",
]


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))


@lru_cache(maxsize=1)
def _scenario_client() -> chromadb.PersistentClient:
    SCENARIO_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(SCENARIO_DB_PATH), settings=Settings(anonymized_telemetry=False))


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


def _classify_text(text: str, fallback: str | None = None) -> str | None:
    lower = text.lower()
    if any(keyword in lower for keyword in TREATMENT_KEYWORDS):
        return "traitements"
    if any(keyword in lower for keyword in COLOR_KEYWORDS):
        return "couleurs"
    if any(keyword in lower for keyword in INDEX_KEYWORDS):
        return "indices"
    if any(keyword in lower for keyword in TYPE_KEYWORDS):
        return "types_verres"
    return fallback


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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

    for document_path in sorted([*folder.glob("*.docx"), *folder.glob("*.txt")]):
        fallback_category = _classify_doc(document_path.name)
        text = _read_docx(document_path) if document_path.suffix.lower() == ".docx" else _read_text(document_path)
        _ingest_chunks(client, document_path, _split_lines(text), fallback_category)


def _ingest_chunks(
    client: chromadb.PersistentClient,
    document_path: Path,
    chunks: list[str],
    fallback_category: str | None,
) -> None:
    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = {name: [] for name in COLLECTIONS}
    for idx, chunk in enumerate(chunks, 1):
        category = _classify_text(chunk, fallback=fallback_category)
        if category is None:
            continue
        cid = f"{document_path.stem}_{idx}"
        grouped[category].append((cid, chunk, {"collection": category, "source": document_path.name}))

    for category, items in grouped.items():
        if not items:
            continue
        coll = client.get_collection(category)
        existing = set(coll.get().get("ids", []))
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for cid, chunk, metadata in items:
            if cid in existing:
                continue
            ids.append(cid)
            docs.append(chunk)
            metas.append(metadata)
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


def get_context(query: str, collection: str, n: int = 3) -> str:
    """Retrieve RAG context for one taxonomy collection.

    The taxonomy DB is the preferred source for tool calling. In this
    workspace it can be empty, so a read-only fallback queries the existing
    scenario collection to keep the tool useful until taxonomy ingestion runs.
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"Collection invalide: {collection}")

    normalized_query = str(query or "").strip()
    if not normalized_query:
        return ""

    client = _client()
    coll = client.get_or_create_collection(collection)
    if coll.count() > 0:
        res = coll.query(
            query_texts=[normalized_query],
            n_results=n,
            where={"collection": collection},
        )
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        return _format_context(docs, metadatas, source="taxonomy")

    try:
        scenario_coll = _scenario_client().get_collection("scenarios_optique")
    except Exception:
        return ""
    if scenario_coll.count() == 0:
        return ""
    res = scenario_coll.query(query_texts=[normalized_query], n_results=n)
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    return _format_context(docs, metadatas, source="scenarios_optique")


def _format_context(docs: list[str], metadatas: list[dict[str, Any] | None], source: str) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, 1):
        meta = metadatas[idx - 1] if idx - 1 < len(metadatas) else None
        meta_text = f" metadata={meta}" if meta else ""
        lines.append(f"[{source} #{idx}]{meta_text}\n{doc}")
    return "\n\n---\n\n".join(lines)
