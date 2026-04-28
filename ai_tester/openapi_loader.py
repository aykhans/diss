"""
OpenAPI sxemini yükləmə və hərəkət fəzasına çevirmə.

Backend-in /openapi.json endpoint-indən sxemi alıb, RL agentinin istifadə
edəcəyi əməliyyatlar siyahısına ayırır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class Operation:
    """Bir API əməliyyatının abstrakt təsviri."""

    op_id: str
    method: str  # GET, POST, PUT, DELETE
    path: str  # /pets/{pet_id}
    path_params: list[str] = field(default_factory=list)
    body_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    summary: str = ""

    def display(self) -> str:
        return f"{self.method} {self.path}"


def fetch_openapi(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Backend-dən OpenAPI sxemini al."""
    r = requests.get(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _resolve_ref(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    """`$ref` referansını `#/components/schemas/X` formatından oxu."""
    parts = ref.lstrip("#/").split("/")
    obj: Any = schema
    for part in parts:
        obj = obj[part]
    return obj


def _inline_refs(node: Any, root: dict[str, Any]) -> Any:
    """Sxem ağacında bütün $ref-ləri yerinə əvəz et."""
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            resolved = _resolve_ref(root, node["$ref"])
            return _inline_refs(resolved, root)
        return {k: _inline_refs(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, root) for v in node]
    return node


def parse_operations(openapi: dict[str, Any]) -> list[Operation]:
    """OpenAPI sxemindən bütün əməliyyatları çıxar."""
    ops: list[Operation] = []
    paths = openapi.get("paths", {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, spec in methods.items():
            if method.upper() not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            if not isinstance(spec, dict):
                continue
            op_id = spec.get("operationId") or f"{method}_{path}"
            path_params = [
                p.get("name", "") for p in spec.get("parameters", []) if p.get("in") == "path"
            ]
            body_schema = None
            req = spec.get("requestBody", {})
            content = req.get("content", {}) if isinstance(req, dict) else {}
            if "application/json" in content:
                body_schema = _inline_refs(content["application/json"].get("schema"), openapi)
            response_schema = None
            for code in ("200", "201", "204"):
                resp = spec.get("responses", {}).get(code, {})
                rcontent = resp.get("content", {}) if isinstance(resp, dict) else {}
                if "application/json" in rcontent:
                    response_schema = _inline_refs(
                        rcontent["application/json"].get("schema"), openapi
                    )
                    break
            ops.append(
                Operation(
                    op_id=op_id,
                    method=method.upper(),
                    path=path,
                    path_params=path_params,
                    body_schema=body_schema,
                    response_schema=response_schema,
                    summary=spec.get("summary", ""),
                )
            )
    return ops
