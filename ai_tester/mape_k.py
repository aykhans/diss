"""
MAPE-K idarəetmə qatı.

IBM Autonomic Computing modelinə əsaslanır:
    Monitor  -> hədəf sistemin OpenAPI və davranış statistikalarını izlə
    Analyze  -> drift hesabatını qur, problemli ssenariləri müəyyən et
    Plan     -> bərpa strategiyasını seç (rule vs LLM, hansı ssenarilərə
                tətbiq etmək)
    Execute  -> bərpa edilmiş ssenarini icra et və nəticəni qiymətləndir
    Knowledge-> bütün dəyişiklikləri SQLite bilik bazasına yaz

Bu modul yuxarıda göstərilən komponentləri (RestApiEnv, KnowledgeBase,
self_healing, concept_drift) əlaqələndirən orkestrator rolunu oynayır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from ai_tester.concept_drift import DriftReport, detect_drift
from ai_tester.knowledge_base import KnowledgeBase
from ai_tester.openapi_loader import fetch_openapi, parse_operations
from ai_tester.self_healing import TestScenario, heal_scenario

logger = logging.getLogger(__name__)


@dataclass
class CycleReport:
    drift_id: int | None
    drift_summary: str
    repaired_count: int
    successful_count: int
    failed_count: int


class MapeKController:
    """Bir backend nümunəsi və bilik bazası ətrafında MAPE-K döngüsü."""

    def __init__(
        self,
        base_url: str,
        kb: KnowledgeBase,
        use_llm: bool = True,
        request_timeout: float = 4.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.kb = kb
        self.use_llm = use_llm
        self.request_timeout = request_timeout
        self._session = requests.Session()

    # ----------------- Monitor -----------------
    def monitor(self, label: str = "") -> tuple[int, dict[str, Any]]:
        """Cari OpenAPI sxemini al və bilik bazasına yaz."""
        openapi = fetch_openapi(self.base_url)
        snap_id = self.kb.save_snapshot(openapi, label=label)
        logger.info("Snapshot saxlanıldı: id=%d label=%r", snap_id, label)
        return snap_id, openapi

    # ----------------- Analyze -----------------
    def analyze(self, old_snap_id: int, new_snap_id: int) -> tuple[int | None, DriftReport]:
        """İki snapshot arasında dəyişikliyi aşkarla və bilik bazasına yaz."""
        old = self.kb.load_snapshot(old_snap_id)
        new = self.kb.load_snapshot(new_snap_id)
        report = detect_drift(old, new)
        if report.is_empty():
            logger.info("Heç bir dəyişiklik aşkar edilmədi.")
            return None, report
        report_json = {
            "new_ops": [o.display() for o in report.new_ops],
            "removed_ops": [o.display() for o in report.removed_ops],
            "renamed_ops": [
                {"from": old.display(), "to": new.display()} for old, new in report.renamed_ops
            ],
            "field_changes": [
                {
                    "container": fc.container,
                    "op_id": fc.op_id,
                    "added": fc.added,
                    "removed": fc.removed,
                    "renamed": fc.renamed,
                }
                for fc in report.field_changes
            ],
        }
        drift_id = self.kb.save_drift(old_snap_id, new_snap_id, report.summary(), report_json)
        return drift_id, report

    # ----------------- Plan + Execute -----------------
    def plan_and_execute(
        self,
        drift_id: int,
        report: DriftReport,
        new_openapi: dict[str, Any],
        scenario_ids: list[int] | None = None,
    ) -> CycleReport:
        """Bütün (və ya verilmiş) ssenariləri bərpa et və icra et."""
        new_ops = parse_operations(new_openapi)
        if scenario_ids is None:
            scenario_ids = [
                s["id"] for s in self.kb.list_scenarios() if s["source"] in ("rl", "manual")
            ]

        ok = 0
        fail = 0
        repaired_count = 0
        for sid in scenario_ids:
            payload = self.kb.load_scenario(sid)
            scenario = TestScenario.from_dict(payload)
            repaired, source = heal_scenario(scenario, report, new_ops, use_llm=self.use_llm)
            new_sid = self.kb.save_scenario(
                name=repaired.name,
                source=f"{source}-repair",
                payload=repaired.to_dict(),
                parent_id=sid,
            )
            run_success, status_codes = self._run_scenario(repaired)
            self.kb.save_run(
                new_sid,
                status_codes,
                run_success,
                notes=f"repaired by {source}",
            )
            self.kb.save_repair(
                drift_id=drift_id,
                original_scenario_id=sid,
                repaired_scenario_id=new_sid,
                source=source,
                success=run_success,
            )
            repaired_count += 1
            if run_success:
                ok += 1
            else:
                fail += 1

        return CycleReport(
            drift_id=drift_id,
            drift_summary=report.summary(),
            repaired_count=repaired_count,
            successful_count=ok,
            failed_count=fail,
        )

    # ----------------- Helper: ssenarini icra et -----------------
    def _run_scenario(self, scenario: TestScenario) -> tuple[bool, list[int]]:
        """Ssenarini icra et və "self-healing uğuru" meyarına görə qiymətləndir.

        Uğur meyarı: sxem-pozulmasına dair heç bir 422 (Unprocessable Entity)
        və ya 5xx olmamalı, eyni zamanda ən azı bir 2xx olmalıdır. 404-lər
        məzmuna aid olduqları üçün (köhnə resource id-ləri yeni mühitdə
        olmaya bilər) sxem-bərpa qiymətləndirməsində nəzərə alınmır.
        """
        codes: list[int] = []
        for step in scenario.steps:
            url = self.base_url + step.path
            for k, v in step.path_args.items():
                url = url.replace("{" + k + "}", str(v))
            try:
                r = self._session.request(
                    step.method,
                    url,
                    json=step.payload if step.payload else None,
                    timeout=self.request_timeout,
                )
                codes.append(r.status_code)
            except requests.RequestException:
                codes.append(599)
        if not codes:
            return False, codes
        no_schema_err = all(c not in (422,) and c < 500 for c in codes)
        any_2xx = any(200 <= c < 300 for c in codes)
        return (no_schema_err and any_2xx), codes

    # ----------------- Yüksək səviyyəli müəssisə döngüsü -----------------
    def run_cycle(self, label: str = "") -> CycleReport | None:
        """Tam M-A-P-E döngüsünü yerinə yetir.

        Köhnə snapshot mövcud deyilsə, yalnız ilkin snapshot yaradılır və
        bərpa addımları ötürülür.
        """
        prev = self.kb.latest_snapshot()
        new_snap_id, new_openapi = self.monitor(label=label)
        if prev is None:
            logger.info("İlkin snapshot saxlanıldı; bərpa addımı atlanır.")
            return None
        old_snap_id, _ = prev
        drift_id, report = self.analyze(old_snap_id, new_snap_id)
        if drift_id is None:
            return None
        return self.plan_and_execute(drift_id, report, new_openapi)
