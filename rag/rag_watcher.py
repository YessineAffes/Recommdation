"""Watcher RAG : surveille rag/documents/, ingere .docx (existant) + .json produit (nouveau)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.rag_builder import COLLECTIONS, _client, ingest_folder

ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "rag" / "documents"


def _product_name(data: dict[str, Any]) -> str:
    return str(data.get("nom") or data.get("nom_produit") or "").strip()


def _infer_collection(data: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in (
            data.get("nom"),
            data.get("nom_produit"),
            data.get("famille"),
            data.get("concept"),
            data.get("description"),
        )
        if value
    ).lower()
    if any(keyword in text for keyword in ("crizal", "traitement", "prevencia", "sapphire", "alize")):
        return "traitements"
    if any(keyword in text for keyword in ("transition", "transitions", "brun", "gris", "photochromique", "solaire")):
        return "couleurs"
    if any(keyword in text for keyword in ("indice", "1.50", "1.60", "1.67", "1.74")):
        return "indices"
    return "types_verres"


def _build_product_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    if data.get("nom_produit"):
        parts.append(f"Produit: {data['nom_produit']}")
    if data.get("concept"):
        parts.append(f"Concept: {data['concept']}")
    plage = data.get("plage_performance")
    if isinstance(plage, dict):
        parts.append(f"Plage de performance: {plage.get('min')} a {plage.get('max')}")
    avantages = data.get("avantages")
    if isinstance(avantages, list):
        parts.extend(f"Avantage: {item}" for item in avantages if str(item).strip())
    elif avantages:
        parts.append(f"Avantages: {avantages}")
    if data.get("recommande_pour"):
        parts.append(f"Recommande pour: {data['recommande_pour']}")
    for reference in data.get("references") or []:
        if not isinstance(reference, dict):
            continue
        parts.append(
            "Reference: "
            f"{reference.get('nom', '')}; plage stock {reference.get('plage_stock', '')}; "
            f"cyl 200 {reference.get('cyl_200', '')}; cyl 100 {reference.get('cyl_100', '')}; "
            f"spherique {reference.get('spherique', '')}"
        )
    for note in data.get("notes") or []:
        if isinstance(note, dict):
            parts.append(f"Note: {note.get('titre', '')} - {note.get('contenu', '')}")

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


def ingest_product_payload(
    data: dict[str, Any],
    *,
    source_name: str = "product_form.json",
    collection_cible: str | None = None,
) -> dict[str, Any]:
    """Ingere un payload produit deja charge, ancien ou nouveau schema."""
    collection = collection_cible or data.get("collection_cible") or _infer_collection(data)
    if collection not in COLLECTIONS:
        return {
            "ok": False,
            "message": f"collection_cible '{collection}' invalide (attendu: {COLLECTIONS})",
        }

    name = _product_name(data)
    if not name:
        return {"ok": False, "message": "Champ 'nom' ou 'nom_produit' obligatoire"}

    text = _build_product_text(data)
    if not text.strip():
        return {"ok": False, "message": "Texte d'ingestion vide"}

    client = _client()
    coll = client.get_or_create_collection(collection)

    stem = Path(source_name).stem
    doc_id = f"product_{stem}"
    metadata = {
        "collection": collection,
        "source": "form",
        "nom": name,
        "famille": str(data.get("famille") or ""),
        "filename": Path(source_name).name,
    }
    existing = set(coll.get().get("ids", []))
    if doc_id in existing:
        coll.update(ids=[doc_id], documents=[text], metadatas=[metadata])
    else:
        coll.add(ids=[doc_id], documents=[text], metadatas=[metadata])

    return {
        "ok": True,
        "collection": collection,
        "id": doc_id,
        "message": f"Produit '{name}' indexe dans '{collection}'",
    }


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

    return ingest_product_payload(data, source_name=fp.name)


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
