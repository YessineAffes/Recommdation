"""Store corrections expert: SQLite local ou PostgreSQL/Supabase en production."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    desc,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Connection, Engine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "corrections.db"

HISTORY_PENDING = "pending"
HISTORY_CORRECTED = "corrected"
HISTORY_VALIDATED = "validated"
HISTORY_STATUSES = {HISTORY_PENDING, HISTORY_CORRECTED, HISTORY_VALIDATED}

metadata = MetaData()

corrections_table = Table(
    "corrections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("state_hash", Text, nullable=False),
    Column("state_json", Text, nullable=False),
    Column("engine_output", Text, nullable=False),
    Column("expert_output", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("justification", Text),
    Column("age", Integer),
    Column("correction_num", Float),
    Column("montage", Text),
    CheckConstraint("status IN ('CORRECT','OVERRIDE')", name="ck_corrections_status"),
)
Index("idx_corr_hash", corrections_table.c.state_hash)
Index("idx_corr_lookup", corrections_table.c.status, corrections_table.c.montage, corrections_table.c.age, corrections_table.c.correction_num)

history_table = Table(
    "recommendation_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    Column("state_hash", Text, nullable=False),
    Column("input_json", Text, nullable=False),
    Column("recommendation_auto", Text, nullable=False),
    Column("recommendation_expert", Text),
    Column("status", Text, nullable=False),
    Column("expert_name", Text),
    Column("notes", Text),
    CheckConstraint("status IN ('pending','corrected','validated')", name="ck_history_status"),
)
Index("idx_history_created", history_table.c.created_at)
Index("idx_history_status", history_table.c.status, history_table.c.updated_at)
Index("idx_history_hash", history_table.c.state_hash)

product_cards_table = Table(
    "product_cards",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    Column("expert_name", Text),
    Column("product_name", Text, nullable=False),
    Column("collection_cible", Text, nullable=False),
    Column("source_file", Text),
    Column("payload_json", Text, nullable=False),
    Column("rag_result_json", Text),
)
Index("idx_product_cards_created", product_cards_table.c.created_at)
Index("idx_product_cards_collection", product_cards_table.c.collection_cible, product_cards_table.c.updated_at)

_CORRECTION_BUCKETS = {
    "<200": 1.00,
    "225-400": 3.10,
    "425-600": 5.10,
    ">625": 7.00,
}


def _correction_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in _CORRECTION_BUCKETS:
        return _CORRECTION_BUCKETS[text]
    try:
        return float(text)
    except ValueError:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    return url


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def hash_state(state: dict[str, Any]) -> str:
    blob = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


class CorrectionsStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB, database_url: str | None = None):
        explicit_db_path = Path(db_path) != DEFAULT_DB
        raw_url = database_url or (None if explicit_db_path else os.getenv("DATABASE_URL"))
        if raw_url:
            self.db_path: Path | None = None
            self.database_url = _normalize_database_url(raw_url)
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = _sqlite_url(self.db_path)

        self.engine = self._create_engine(self.database_url)
        self._init_schema()

    def _create_engine(self, database_url: str) -> Engine:
        kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        return create_engine(database_url, **kwargs)

    def _init_schema(self) -> None:
        metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            if self.database_url.startswith("sqlite"):
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            self._migrate_legacy_corrections(conn)

    def _migrate_legacy_corrections(self, conn: Connection) -> None:
        rows = conn.execute(select(corrections_table)).mappings().all()
        for row in rows:
            history_status = HISTORY_VALIDATED if row["status"] == "CORRECT" else HISTORY_CORRECTED
            exists = conn.execute(
                select(history_table.c.id).where(
                    and_(
                        history_table.c.state_hash == row["state_hash"],
                        or_(
                            history_table.c.created_at == row["timestamp"],
                            and_(
                                history_table.c.recommendation_auto == row["engine_output"],
                                history_table.c.recommendation_expert == row["expert_output"],
                                history_table.c.status == history_status,
                            ),
                        ),
                    )
                )
            ).first()
            if exists:
                continue
            conn.execute(
                insert(history_table).values(
                    created_at=row["timestamp"],
                    updated_at=row["timestamp"],
                    state_hash=row["state_hash"],
                    input_json=row["state_json"],
                    recommendation_auto=row["engine_output"],
                    recommendation_expert=row["expert_output"],
                    status=history_status,
                    expert_name=None,
                    notes=row["justification"],
                )
            )

    def _decode_history_row(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        for raw_key, parsed_key in (
            ("input_json", "input_data"),
            ("recommendation_auto", "recommendation_auto_data"),
            ("recommendation_expert", "recommendation_expert_data"),
        ):
            raw = data.get(raw_key)
            if raw in (None, ""):
                data[parsed_key] = None
                continue
            try:
                data[parsed_key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data[parsed_key] = raw
        return data

    def _decode_product_row(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        payload_raw = data.get("payload_json")
        rag_raw = data.get("rag_result_json")
        try:
            data["payload_data"] = json.loads(payload_raw) if payload_raw else None
        except (json.JSONDecodeError, TypeError):
            data["payload_data"] = payload_raw
        try:
            data["rag_result_data"] = json.loads(rag_raw) if rag_raw else None
        except (json.JSONDecodeError, TypeError):
            data["rag_result_data"] = rag_raw
        return data

    def add_product_card(
        self,
        payload: dict[str, Any],
        collection_cible: str,
        expert_name: str | None = None,
        source_file: str | None = None,
        rag_result: dict[str, Any] | None = None,
    ) -> int:
        product_name = str(payload.get("nom_produit") or payload.get("nom") or "").strip()
        if not product_name:
            raise ValueError("nom_produit (ou nom) est obligatoire pour enregistrer la fiche produit")

        timestamp = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(product_cards_table).values(
                    created_at=timestamp,
                    updated_at=timestamp,
                    expert_name=expert_name,
                    product_name=product_name,
                    collection_cible=str(collection_cible),
                    source_file=source_file,
                    payload_json=_json_dumps(payload),
                    rag_result_json=_json_dumps(rag_result) if rag_result is not None else None,
                )
            )
            return int(result.inserted_primary_key[0])

    def get_product_card(self, record_id: int) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(product_cards_table).where(product_cards_table.c.id == int(record_id))
            ).mappings().first()
        return self._decode_product_row(row) if row is not None else None

    def list_product_cards(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(product_cards_table).order_by(desc(product_cards_table.c.created_at)).limit(int(limit))
            ).mappings().all()
        return [self._decode_product_row(row) for row in rows]

    def create_history(
        self,
        state: dict[str, Any],
        recommendation_auto: dict[str, Any],
        expert_name: str | None = None,
    ) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(history_table).values(
                    created_at=timestamp,
                    updated_at=timestamp,
                    state_hash=hash_state(state),
                    input_json=_json_dumps(state),
                    recommendation_auto=_json_dumps(recommendation_auto),
                    recommendation_expert=None,
                    status=HISTORY_PENDING,
                    expert_name=expert_name,
                    notes=None,
                )
            )
            return int(result.inserted_primary_key[0])

    def update_history(
        self,
        record_id: int,
        recommendation_expert: dict[str, Any],
        status: str,
        expert_name: str | None = None,
        notes: str | None = None,
    ) -> None:
        if status not in {HISTORY_CORRECTED, HISTORY_VALIDATED}:
            raise ValueError(f"Statut historique invalide: {status}")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(history_table)
                .where(history_table.c.id == int(record_id))
                .values(
                    updated_at=timestamp,
                    recommendation_expert=_json_dumps(recommendation_expert),
                    status=status,
                    expert_name=expert_name,
                    notes=notes,
                )
            )
            if result.rowcount == 0:
                raise KeyError(f"Historique introuvable: id={record_id}")

    def get_history(self, record_id: int) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(history_table).where(history_table.c.id == int(record_id))).mappings().first()
        return self._decode_history_row(row) if row is not None else None

    def list_history(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        stmt = select(history_table).order_by(desc(history_table.c.created_at)).limit(int(limit))
        if status:
            if status not in HISTORY_STATUSES:
                raise ValueError(f"Statut historique invalide: {status}")
            stmt = stmt.where(history_table.c.status == status)
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._decode_history_row(row) for row in rows]

    def add(
        self,
        state: dict[str, Any],
        engine_output: dict[str, Any],
        expert_output: dict[str, Any],
        status: str,
        justification: str | None = None,
        sync_history: bool = True,
    ) -> int:
        assert status in ("CORRECT", "OVERRIDE")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(corrections_table).values(
                    timestamp=timestamp,
                    state_hash=hash_state(state),
                    state_json=_json_dumps(state),
                    engine_output=_json_dumps(engine_output),
                    expert_output=_json_dumps(expert_output),
                    status=status,
                    justification=justification,
                    age=int(state.get("Q1_age")) if state.get("Q1_age") is not None else None,
                    correction_num=_correction_to_float(state.get("Q3_correction_total")),
                    montage=state.get("Q2_montage"),
                )
            )
            correction_id = int(result.inserted_primary_key[0])
        if sync_history:
            history_id = self.create_history(state, engine_output)
            self.update_history(
                history_id,
                expert_output,
                HISTORY_VALIDATED if status == "CORRECT" else HISTORY_CORRECTED,
                notes=justification,
            )
        return correction_id

    def find_override(
        self,
        state: dict[str, Any],
        age_tol: int = 2,
        correction_tol: float = 0.25,
    ) -> dict[str, Any] | None:
        """Retourne le dernier expert_output OVERRIDE proche, ou None."""
        age = state.get("Q1_age")
        montage = state.get("Q2_montage")
        correction = _correction_to_float(state.get("Q3_correction_total"))
        if age is None or montage is None or correction is None:
            return None

        stmt = (
            select(corrections_table.c.expert_output)
            .where(
                and_(
                    corrections_table.c.status == "OVERRIDE",
                    corrections_table.c.montage == montage,
                    func.abs(corrections_table.c.age - int(age)) <= age_tol,
                    func.abs(corrections_table.c.correction_num - float(correction)) <= correction_tol,
                )
            )
            .order_by(desc(corrections_table.c.timestamp))
            .limit(1)
        )
        with self.engine.begin() as conn:
            row = conn.execute(stmt).first()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(corrections_table).order_by(desc(corrections_table.c.timestamp)).limit(int(limit))
            ).mappings().all()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self.engine.begin() as conn:
            value = conn.execute(select(func.count()).select_from(corrections_table)).scalar_one()
        return int(value)

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(corrections_table))
            conn.execute(delete(history_table))
            conn.execute(delete(product_cards_table))
