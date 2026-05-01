"""Watcher RAG : surveille rag/documents/, ingere .docx (existant) + .json produit (nouveau)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.rag_builder import COLLECTIONS, _client, ingest_folder

ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "rag" / "documents"


def _build_product_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    if data.get("nom"):
        parts.append(f"Produit: {data['nom']}")
    if data.get("famille"):
        parts.append(f"Famille: {data['famille']}")
    if data.get("sous_famille"):
        parts.append(f"Sous-famille: {data['sous_famille']}")
    if data.get("description"):
        parts.append(f"Description: {data['description']}")
    if data.get("indications"):
        parts.append(f"Indications: {data['indications']}")
    if data.get("contre_indications"):
        parts.append(f"Contre-indications: {data['contre_indications']}")
    if data.get("avantages"):
        parts.append(f"Avantages: {data['avantages']}")
    if data.get("comparaison_precedent"):
        parts.append(f"Comparaison: {data['comparaison_precedent']}")
    return "\n".join(parts)


def ingest_product_json(filepath: str | Path) -> dict[str, Any]:
    """Ingere un fichier produit JSON dans la collection ChromaDB cible.

    Retourne {'ok': bool, 'collection': str, 'id': str, 'message': str}.
    """
    fp = Path(filepath)
    if not fp.exists():
        return {"ok": False, "message": f"Fichier introuvable: {fp}"}

    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"ok": False, "message": f"JSON invalide: {e}"}

    collection = data.get("collection_cible")
    if collection not in COLLECTIONS:
        return {
            "ok": False,
            "message": f"collection_cible '{collection}' invalide (attendu: {COLLECTIONS})",
        }
    if not data.get("nom"):
        return {"ok": False, "message": "Champ 'nom' obligatoire"}

    text = _build_product_text(data)
    if not text.strip():
        return {"ok": False, "message": "Texte d'ingestion vide"}

    client = _client()
    coll = client.get_or_create_collection(collection)

    doc_id = f"product_{fp.stem}"
    metadata = {
        "collection": collection,
        "source": "form",
        "nom": data.get("nom", ""),
        "famille": data.get("famille", ""),
        "filename": fp.name,
    }
    # upsert : si l'id existe, on remplace
    existing = set(coll.get().get("ids", []))
    if doc_id in existing:
        coll.update(ids=[doc_id], documents=[text], metadatas=[metadata])
    else:
        coll.add(ids=[doc_id], documents=[text], metadatas=[metadata])

    return {
        "ok": True,
        "collection": collection,
        "id": doc_id,
        "message": f"Produit '{data['nom']}' indexe dans '{collection}'",
    }


def watch_once(folder: Path | None = None) -> dict[str, Any]:
    """Scan unique : ingere les .docx (via rag_builder) + tous les .json produit."""
    folder = folder or DOCS_PATH
    folder.mkdir(parents=True, exist_ok=True)

    ingest_folder(folder)  # docx existants

    results: list[dict[str, Any]] = []
    for jf in folder.glob("*.json"):
        results.append({"file": jf.name, **ingest_product_json(jf)})

    return {"docx_scanned": True, "json_results": results}


if __name__ == "__main__":
    out = watch_once()
    print(json.dumps(out, indent=2, ensure_ascii=False))
