"""SQLite store pour corrections expert (CRUD + lookup approximatif)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "corrections.db"

HISTORY_PENDING = "pending"
HISTORY_CORRECTED = "corrected"
HISTORY_VALIDATED = "validated"
HISTORY_STATUSES = {HISTORY_PENDING, HISTORY_CORRECTED, HISTORY_VALIDATED}

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    state_hash      TEXT    NOT NULL,
    state_json      TEXT    NOT NULL,
    engine_output   TEXT    NOT NULL,
    expert_output   TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK(status IN ('CORRECT','OVERRIDE')),
    justification   TEXT,
    age             INTEGER,
    correction_num  REAL,
    montage         TEXT
);
CREATE INDEX IF NOT EXISTS idx_corr_hash   ON corrections(state_hash);
CREATE INDEX IF NOT EXISTS idx_corr_lookup ON corrections(status, montage, age, correction_num);

CREATE TABLE IF NOT EXISTS recommendation_history (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL,
    state_hash            TEXT    NOT NULL,
    input_json            TEXT    NOT NULL,
    recommendation_auto   TEXT    NOT NULL,
    recommendation_expert TEXT,
    status                TEXT    NOT NULL CHECK(status IN ('pending','corrected','validated')),
    expert_name           TEXT,
    notes                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_created ON recommendation_history(created_at);
CREATE INDEX IF NOT EXISTS idx_history_status  ON recommendation_history(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_history_hash    ON recommendation_history(state_hash);
"""

# Conversion bucket Q3 -> mediane numerique (en dioptries)
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
    s = str(value).strip()
    if s in _CORRECTION_BUCKETS:
        return _CORRECTION_BUCKETS[s]
    try:
        return float(s)
    except ValueError:
        return None


def hash_state(state: dict[str, Any]) -> str:
    blob = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


class CorrectionsStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)
            self._migrate_legacy_corrections(c)

    def _migrate_legacy_corrections(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO recommendation_history
                (created_at, updated_at, state_hash, input_json, recommendation_auto,
                 recommendation_expert, status, expert_name, notes)
            SELECT
                timestamp,
                timestamp,
                state_hash,
                state_json,
                engine_output,
                expert_output,
                CASE status WHEN 'CORRECT' THEN 'validated' ELSE 'corrected' END,
                NULL,
                justification
            FROM corrections old
            WHERE NOT EXISTS (
                SELECT 1 FROM recommendation_history hist
                WHERE hist.state_hash = old.state_hash
                  AND hist.created_at = old.timestamp
            )
            """
        )

    def _decode_history_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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

    def create_history(
        self,
        state: dict[str, Any],
        recommendation_auto: dict[str, Any],
        expert_name: str | None = None,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO recommendation_history
                    (created_at, updated_at, state_hash, input_json, recommendation_auto,
                     recommendation_expert, status, expert_name, notes)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts,
                    ts,
                    hash_state(state),
                    json.dumps(state, default=str),
                    json.dumps(recommendation_auto, default=str),
                    None,
                    HISTORY_PENDING,
                    expert_name,
                    None,
                ),
            )
            return int(cur.lastrowid)

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
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE recommendation_history
                   SET updated_at=?, recommendation_expert=?, status=?, expert_name=?, notes=?
                 WHERE id=?
                """,
                (
                    ts,
                    json.dumps(recommendation_expert, default=str),
                    status,
                    expert_name,
                    notes,
                    int(record_id),
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Historique introuvable: id={record_id}")

    def get_history(self, record_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM recommendation_history WHERE id=?", (int(record_id),))
            row = cur.fetchone()
        return self._decode_history_row(row) if row is not None else None

    def list_history(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if status:
            if status not in HISTORY_STATUSES:
                raise ValueError(f"Statut historique invalide: {status}")
            where = "WHERE status=?"
            params.append(status)
        params.append(int(limit))
        with self._conn() as c:
            cur = c.execute(
                f"SELECT * FROM recommendation_history {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            return [self._decode_history_row(r) for r in cur.fetchall()]

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
        ts = datetime.now(timezone.utc).isoformat()
        row = (
            ts,
            hash_state(state),
            json.dumps(state, default=str),
            json.dumps(engine_output, default=str),
            json.dumps(expert_output, default=str),
            status,
            justification,
            int(state.get("Q1_age")) if state.get("Q1_age") is not None else None,
            _correction_to_float(state.get("Q3_correction_total")),
            state.get("Q2_montage"),
        )
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO corrections
                   (timestamp,state_hash,state_json,engine_output,expert_output,
                    status,justification,age,correction_num,montage)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            correction_id = int(cur.lastrowid)
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
        corr = _correction_to_float(state.get("Q3_correction_total"))
        if age is None or montage is None or corr is None:
            return None

        with self._conn() as c:
            cur = c.execute(
                """SELECT expert_output FROM corrections
                    WHERE status='OVERRIDE'
                      AND montage=?
                      AND ABS(age - ?) <= ?
                      AND ABS(correction_num - ?) <= ?
                    ORDER BY timestamp DESC LIMIT 1""",
                (montage, int(age), age_tol, float(corr), correction_tol),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["expert_output"])
        except (json.JSONDecodeError, TypeError):
            return None

    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT * FROM corrections ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM corrections").fetchone()[0])

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM corrections")
