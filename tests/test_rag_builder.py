from __future__ import annotations


def test_ingest_txt_chunks_into_taxonomy_collections(tmp_path, monkeypatch):
    import rag.rag_builder as rb

    monkeypatch.setattr(rb, "DB_PATH", tmp_path / "chroma_taxonomy")
    rb._client.cache_clear()

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "dump.txt").write_text(
        "Simple foyer et Eyezen sont des types de verres.\n"
        "Indice 1.60 minimum pour montage perce.\n"
        "Crizal Prevencia protege de la lumiere bleue.\n"
        "Transitions Gen S est une solution photochromique.\n",
        encoding="utf-8",
    )

    rb.ingest_folder(docs, reset=True)

    client = rb._client()
    assert client.get_collection("types_verres").count() == 1
    assert client.get_collection("indices").count() == 1
    assert client.get_collection("traitements").count() == 1
    assert client.get_collection("couleurs").count() == 1

    context = rb.get_context("Crizal Prevencia", "traitements", n=1)
    assert "Crizal Prevencia" in context
    assert "taxonomy" in context

    rb._client.cache_clear()