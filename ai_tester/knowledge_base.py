"""
Bilik bazası: SQLite-əsaslı saxlayıcı.

Saxlanılan məlumatlar:
- snapshots: zamanın hər anında alınan OpenAPI sxemləri
- scenarios: test ssenariləri (orijinal və bərpa edilmiş)
- runs: test ssenarisinin icra nəticələri
- drift_events: aşkar edilmiş konsept dəyişikliyi hesabatları
- repairs: bərpa cəhdlərinin nəticələri (uğurlu/uğursuz, mənbə)

Bu məlumatlar həm reproducability, həm də gələcək təlim/qiymətləndirmə üçün
istifadə olunur.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    label TEXT,
    openapi_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    name TEXT,
    source TEXT,            -- "rl" | "rule-repair" | "llm-repair" | "manual"
    parent_id INTEGER,      -- bərpa edilmişsə orijinal ssenarinin id-si
    payload_json TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    scenario_id INTEGER NOT NULL,
    status_codes TEXT NOT NULL,    -- JSON list
    success INTEGER NOT NULL,      -- 0/1
    notes TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    old_snapshot_id INTEGER NOT NULL,
    new_snapshot_id INTEGER NOT NULL,
    summary TEXT,
    report_json TEXT,
    FOREIGN KEY(old_snapshot_id) REFERENCES snapshots(id),
    FOREIGN KEY(new_snapshot_id) REFERENCES snapshots(id)
);

CREATE TABLE IF NOT EXISTS repairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    drift_id INTEGER NOT NULL,
    original_scenario_id INTEGER NOT NULL,
    repaired_scenario_id INTEGER,
    source TEXT,                   -- "llm" | "rule"
    success INTEGER NOT NULL,
    FOREIGN KEY(drift_id) REFERENCES drift_events(id),
    FOREIGN KEY(original_scenario_id) REFERENCES scenarios(id),
    FOREIGN KEY(repaired_scenario_id) REFERENCES scenarios(id)
);
"""


class KnowledgeBase:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self.path)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # ---------- snapshots ----------
    def save_snapshot(self, openapi: dict[str, Any], label: str = "") -> int:
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO snapshots (ts, label, openapi_json) VALUES (?, ?, ?)",
                (time.time(), label, json.dumps(openapi)),
            )
            return int(cur.lastrowid)

    def load_snapshot(self, snap_id: int) -> dict[str, Any]:
        with self._conn() as cx:
            row = cx.execute("SELECT openapi_json FROM snapshots WHERE id=?", (snap_id,)).fetchone()
            if not row:
                raise KeyError(f"snapshot {snap_id} tapılmadı")
            return json.loads(row["openapi_json"])

    def latest_snapshot(self) -> tuple[int, dict[str, Any]] | None:
        with self._conn() as cx:
            row = cx.execute(
                "SELECT id, openapi_json FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            return int(row["id"]), json.loads(row["openapi_json"])

    # ---------- scenarios ----------
    def save_scenario(
        self,
        name: str,
        source: str,
        payload: dict[str, Any],
        parent_id: int | None = None,
    ) -> int:
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO scenarios (ts, name, source, parent_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), name, source, parent_id, json.dumps(payload)),
            )
            return int(cur.lastrowid)

    def load_scenario(self, scenario_id: int) -> dict[str, Any]:
        with self._conn() as cx:
            row = cx.execute(
                "SELECT payload_json FROM scenarios WHERE id=?", (scenario_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"scenario {scenario_id} tapılmadı")
            return json.loads(row["payload_json"])

    def list_scenarios(self) -> list[dict[str, Any]]:
        with self._conn() as cx:
            rows = cx.execute(
                "SELECT id, ts, name, source, parent_id FROM scenarios ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- runs ----------
    def save_run(
        self,
        scenario_id: int,
        status_codes: list[int],
        success: bool,
        notes: str = "",
    ) -> int:
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO runs (ts, scenario_id, status_codes, success, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    time.time(),
                    scenario_id,
                    json.dumps(status_codes),
                    1 if success else 0,
                    notes,
                ),
            )
            return int(cur.lastrowid)

    # ---------- drift / repairs ----------
    def save_drift(
        self,
        old_snap: int,
        new_snap: int,
        summary: str,
        report: dict[str, Any],
    ) -> int:
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO drift_events (ts, old_snapshot_id, new_snapshot_id, "
                "summary, report_json) VALUES (?, ?, ?, ?, ?)",
                (time.time(), old_snap, new_snap, summary, json.dumps(report)),
            )
            return int(cur.lastrowid)

    def save_repair(
        self,
        drift_id: int,
        original_scenario_id: int,
        repaired_scenario_id: int | None,
        source: str,
        success: bool,
    ) -> int:
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO repairs (ts, drift_id, original_scenario_id, "
                "repaired_scenario_id, source, success) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    drift_id,
                    original_scenario_id,
                    repaired_scenario_id,
                    source,
                    1 if success else 0,
                ),
            )
            return int(cur.lastrowid)

    # ---------- statistika ----------
    def stats(self) -> dict[str, Any]:
        with self._conn() as cx:

            def count(table: str) -> int:
                row = cx.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                return int(row["n"])

            success_rate = 0.0
            row = cx.execute("SELECT AVG(success) AS r FROM repairs").fetchone()
            if row and row["r"] is not None:
                success_rate = float(row["r"])
            return {
                "snapshots": count("snapshots"),
                "scenarios": count("scenarios"),
                "runs": count("runs"),
                "drift_events": count("drift_events"),
                "repairs": count("repairs"),
                "repair_success_rate": success_rate,
            }
