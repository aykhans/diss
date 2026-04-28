"""
RL agentinin episode tarixçəsindən test ssenariləri qurmaq.

Hər episode əslində bir test ssenarisi kimi şərh edilə bilər: agent ardıcıl
hərəkətlər (API çağırışları) edib mükafatlar toplayıb. Uğurlu (2xx) qaytaran
addımları seçib, onlardan ən maraqlı ssenariləri ekstraktə edirik.

Maraq meyarı: ssenarinin ən azı bir 201 (Created) və ya yeni endpoint cütü
ehtiva etməsidir.
"""

from __future__ import annotations

from collections.abc import Iterable

from ai_tester.environment import StepResult
from ai_tester.self_healing import TestScenario, TestStep


def _is_successful(code: int) -> bool:
    return 200 <= code < 300


def steps_from_history(history: Iterable[StepResult]) -> list[TestStep]:
    """Yalnız uğurlu addımları (2xx) ssenari addımlarına çevir."""
    out: list[TestStep] = []
    for h in history:
        if not _is_successful(h.status_code):
            continue
        out.append(
            TestStep(
                method=h.operation.method,
                path=h.operation.path,
                path_args=dict(h.request_path_args),
                payload=dict(h.request_payload),
            )
        )
    return out


def scenario_from_history(
    name: str,
    history: Iterable[StepResult],
) -> TestScenario | None:
    """Episode tarixçəsindən bir TestScenario qur. Uğurlu addım yoxdursa None."""
    history = list(history)
    steps = steps_from_history(history)
    if not steps:
        return None
    expected = [h.status_code for h in history if _is_successful(h.status_code)]
    return TestScenario(
        name=name,
        steps=steps,
        expected_status_codes=expected,
    )


def extract_top_scenarios(
    histories: list[list[StepResult]],
    max_n: int = 5,
) -> list[TestScenario]:
    """Bir neçə episode-un tarixçələrindən ən qiymətli ssenariləri çıxar.

    Qiymət = uğurlu addımların sayı + 2 × unikal endpoint sayı.
    """
    candidates: list[tuple[float, TestScenario]] = []
    for i, h in enumerate(histories):
        sc = scenario_from_history(f"episode_{i}", h)
        if sc is None:
            continue
        unique_endpoints = len({(s.method, s.path) for s in sc.steps})
        score = len(sc.steps) + 2 * unique_endpoints
        candidates.append((score, sc))
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [sc for _, sc in candidates[:max_n]]
