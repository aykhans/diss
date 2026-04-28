"""
RL mühiti: Gymnasium-uyğun mühit, REST API üzərində qarşılıqlı təsir təmin edir.

Vəziyyət (observation): cari sessiya statistikası, yəni son N sorğunun status
kodları, indiyə qədər çağırılmış əməliyyatların sayı, autentifikasiya
statusu (sadə formada).

Hərəkət (action): mümkün API əməliyyatlarından birinin seçilməsi. Hər addımda
agent bir əməliyyat seçir, mühit həmin əməliyyatı parametrləri ilə icra edir
və mükafat qaytarır.

Mükafat:
    +1.0 ilk dəfə görülən 2xx (uğurlu) cavab, coverage artımı kimi
    +0.3 təkrar 2xx
    -0.5 5xx (server xətası)
    -0.05 4xx (müştəri xətası, ümumi olaraq agentin günahı sayılır)
    +1.5 yeni status kodu × yeni endpoint cütlüyü ilk dəfə görüldükdə
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

from ai_tester.openapi_loader import Operation, fetch_openapi, parse_operations
from ai_tester.value_pool import ValuePool, random_value_for_schema

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    operation: Operation
    request_url: str
    status_code: int
    response_body: Any
    is_new_combo: bool
    request_payload: dict[str, Any] = field(default_factory=dict)
    request_path_args: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RestApiEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_url: str,
        episode_length: int = 30,
        observation_window: int = 5,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.episode_length = episode_length
        self.observation_window = observation_window

        # OpenAPI sxemini yüklə və əməliyyatları parse et
        self.openapi = fetch_openapi(self.base_url)
        self.operations: list[Operation] = parse_operations(self.openapi)
        if not self.operations:
            raise RuntimeError("OpenAPI-də heç bir əməliyyat tapılmadı.")

        n_ops = len(self.operations)
        self.action_space = spaces.Discrete(n_ops)

        # Müşahidə fəzası: son N sorğunun status sinfi (0=hələ yoxdur, 1=2xx,
        # 2=4xx, 3=5xx) + son əməliyyatın indeksi + sınanmış əməliyyatların
        # nisbəti + 2xx coverage nisbəti
        self.observation_space = spaces.Dict(
            {
                "recent_status_classes": spaces.MultiDiscrete([4] * observation_window),
                "last_op_idx": spaces.Discrete(n_ops + 1),  # +1 = "hələ heç biri"
                "ops_tried_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "success_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self._session = requests.Session()
        self._reset_internal()

    # ------------------------------------------------------------------
    def _reset_internal(self) -> None:
        self._step_count = 0
        self._recent_status: list[int] = []  # 4-sinifli kodlar
        self._tried_ops: set[int] = set()
        self._success_combos: set[tuple[int, int]] = set()  # (op_idx, status_code)
        self._last_op_idx: int | None = None
        self._value_pool = ValuePool()
        self._history: list[StepResult] = []

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._observation(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        op = self.operations[int(action)]
        result = self._execute(op)
        self._history.append(result)
        self._step_count += 1
        self._tried_ops.add(int(action))
        self._last_op_idx = int(action)

        # Mükafatı hesabla
        cls = _status_class(result.status_code)
        self._recent_status.append(cls)
        self._recent_status = self._recent_status[-self.observation_window :]

        reward = 0.0
        combo = (int(action), result.status_code)
        is_new_combo = combo not in self._success_combos
        if cls == 1:  # 2xx
            reward += 1.0 if is_new_combo else 0.3
            if is_new_combo:
                reward += 0.5  # bonus for new (op, status) cüt
            self._success_combos.add(combo)
            try:
                if result.response_body is not None:
                    self._value_pool.absorb(result.response_body)
            except Exception:
                pass
        elif cls == 3:  # 5xx
            reward -= 0.5
        elif cls == 2:  # 4xx
            reward -= 0.05

        result.is_new_combo = is_new_combo

        terminated = False
        truncated = self._step_count >= self.episode_length
        info = {
            "operation": op.display(),
            "status_code": result.status_code,
            "is_new_combo": is_new_combo,
            "n_unique_combos": len(self._success_combos),
        }
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _observation(self) -> dict[str, Any]:
        recent = (self._recent_status + [0] * self.observation_window)[: self.observation_window]
        n_ops = len(self.operations)
        ops_tried = len(self._tried_ops) / max(1, n_ops)
        successes = sum(1 for s in self._recent_status if s == 1)
        success_ratio = successes / max(1, len(self._recent_status))
        return {
            "recent_status_classes": np.array(recent, dtype=np.int64),
            "last_op_idx": (self._last_op_idx + 1) if self._last_op_idx is not None else 0,
            "ops_tried_ratio": np.array([ops_tried], dtype=np.float32),
            "success_ratio": np.array([success_ratio], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    def _build_path(self, op: Operation) -> tuple[str, dict[str, Any]]:
        path = op.path
        path_args: dict[str, Any] = {}
        for p in op.path_params:
            # Üstünlük: real cavabdan götürülmüş id
            val = self._value_pool.pick(p)
            if val is None:
                # Sxem məhdudiyyəti yoxdur, sadə random uuid kimi
                val = self._value_pool.pick("id")
            if val is None:
                val = "00000000-0000-0000-0000-000000000000"
            path_args[p] = val
            path = path.replace("{" + p + "}", str(val))
        return path, path_args

    def _build_payload(self, op: Operation) -> dict[str, Any]:
        if not op.body_schema:
            return {}
        # Pydantic obyektini sxem nümunəsi kimi şərh edirik
        payload = random_value_for_schema(op.body_schema)
        if isinstance(payload, dict):
            return payload
        return {}

    def _execute(self, op: Operation) -> StepResult:
        path, path_args = self._build_path(op)
        url = self.base_url + path
        payload = self._build_payload(op) if op.method in {"POST", "PUT", "PATCH"} else {}

        try:
            r = self._session.request(
                op.method,
                url,
                json=payload if payload else None,
                timeout=4.0,
            )
            try:
                body = r.json()
            except ValueError:
                body = r.text
            return StepResult(
                operation=op,
                request_url=url,
                status_code=r.status_code,
                response_body=body,
                is_new_combo=False,
                request_payload=payload,
                request_path_args=path_args,
            )
        except requests.RequestException as exc:
            logger.warning("Request failed: %s %s -> %s", op.method, url, exc)
            return StepResult(
                operation=op,
                request_url=url,
                status_code=599,
                response_body=None,
                is_new_combo=False,
                request_payload=payload,
                request_path_args=path_args,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    def render(self) -> None:  # pragma: no cover
        if not self._history:
            print("(boş)")
            return
        last = self._history[-1]
        print(
            f"step={self._step_count} {last.operation.display()} "
            f"-> {last.status_code} new={last.is_new_combo}"
        )

    def history(self) -> list[StepResult]:
        return list(self._history)

    def coverage_summary(self) -> dict[str, Any]:
        n_ops = len(self.operations)
        return {
            "ops_total": n_ops,
            "ops_tried": len(self._tried_ops),
            "unique_2xx_combos": len(self._success_combos),
            "ops_coverage": len(self._tried_ops) / max(1, n_ops),
        }


def _status_class(code: int) -> int:
    if code == 0 or code >= 600:
        return 0
    if 200 <= code < 300:
        return 1
    if 400 <= code < 500:
        return 2
    if 500 <= code < 600:
        return 3
    return 0
