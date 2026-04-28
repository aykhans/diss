"""
Konsept dəyişməsi (concept drift) detektoru.

İki müxtəlif zaman anında alınmış OpenAPI sxemlərini müqayisə edir və
aşağıdakı dəyişiklikləri aşkarlayır:
- silinmiş və ya yeni endpoint-lər
- yenidən adlandırılmış path-lər (heuristik uyğunlaşdırma)
- request body sxemində yeni/silinmiş/yenidən adlandırılmış sahələr
- response sxemində eyni dəyişikliklər
- HTTP metodu dəyişikliyi

Çıxış formatı bilik bazasına və self-healing modulu üçün təlimat kimi
istifadə edilə bilər.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_tester.openapi_loader import Operation, parse_operations


@dataclass
class FieldChange:
    container: str  # "request_body" | "response"
    op_id: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class DriftReport:
    new_ops: list[Operation] = field(default_factory=list)
    removed_ops: list[Operation] = field(default_factory=list)
    renamed_ops: list[tuple[Operation, Operation]] = field(default_factory=list)
    field_changes: list[FieldChange] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.new_ops or self.removed_ops or self.renamed_ops or self.field_changes)

    def summary(self) -> str:
        lines = []
        if self.removed_ops:
            lines.append(f"Silinmiş əməliyyatlar: {len(self.removed_ops)}")
            for op in self.removed_ops:
                lines.append(f"  - {op.display()}")
        if self.new_ops:
            lines.append(f"Yeni əməliyyatlar: {len(self.new_ops)}")
            for op in self.new_ops:
                lines.append(f"  + {op.display()}")
        if self.renamed_ops:
            lines.append(f"Yenidən adlandırılmış: {len(self.renamed_ops)}")
            for old, new in self.renamed_ops:
                lines.append(f"  ~ {old.display()} -> {new.display()}")
        if self.field_changes:
            lines.append(f"Sahə dəyişiklikləri: {len(self.field_changes)}")
            for fc in self.field_changes:
                if fc.added:
                    lines.append(f"  + [{fc.container}] {fc.op_id}: əlavə {fc.added}")
                if fc.removed:
                    lines.append(f"  - [{fc.container}] {fc.op_id}: silindi {fc.removed}")
                if fc.renamed:
                    lines.append(f"  ~ [{fc.container}] {fc.op_id}: {fc.renamed}")
        if not lines:
            return "(dəyişiklik aşkar edilmədi)"
        return "\n".join(lines)


def _fields_of(schema: dict[str, Any] | None) -> set[str]:
    if not schema or not isinstance(schema, dict):
        return set()
    if schema.get("type") == "object" and "properties" in schema:
        return set(schema["properties"].keys())
    return set()


def _heuristic_rename(removed: list[str], added: list[str]) -> list[tuple[str, str]]:
    """Sadə heuristika: silinmiş və əlavə olunmuş sahələri prefiks/postfiks
    və ya genişlənmə əsasında uyğunlaşdırmağa çalışır.
    Məsələn `species` → `kind` üçün uyğunlaşmır, lakin `id` → `pet_id` üçün
    uyğunlaşır.
    """
    pairs: list[tuple[str, str]] = []
    a_used: set[str] = set()
    for r in removed:
        for a in added:
            if a in a_used:
                continue
            if r in a or a in r:  # alt-sətir uyğunluğu
                pairs.append((r, a))
                a_used.add(a)
                break
    return pairs


def diff_operations(old: list[Operation], new: list[Operation]) -> DriftReport:
    """İki əməliyyat siyahısını müqayisə et."""
    report = DriftReport()

    old_keys = {(o.method, o.path): o for o in old}
    new_keys = {(o.method, o.path): o for o in new}

    common = old_keys.keys() & new_keys.keys()
    only_old = old_keys.keys() - new_keys.keys()
    only_new = new_keys.keys() - old_keys.keys()

    # Path renames: oxşar method + suffix uyğunluğu
    only_old_list = [old_keys[k] for k in only_old]
    only_new_list = [new_keys[k] for k in only_new]
    matched_old: set[int] = set()
    matched_new: set[int] = set()
    for i, o in enumerate(only_old_list):
        for j, n in enumerate(only_new_list):
            if j in matched_new:
                continue
            if o.method != n.method:
                continue
            # path suffix uyğunluğu (ən azı bir parametr ilə)
            o_tail = o.path.split("/")[-1]
            n_tail = n.path.split("/")[-1]
            if o_tail == n_tail and o_tail != "":
                report.renamed_ops.append((o, n))
                matched_old.add(i)
                matched_new.add(j)
                break

    for i, o in enumerate(only_old_list):
        if i not in matched_old:
            report.removed_ops.append(o)
    for j, n in enumerate(only_new_list):
        if j not in matched_new:
            report.new_ops.append(n)

    # Sahə dəyişikliklərini həm common, həm də renamed əməliyyatlar üçün yoxla
    pairs: list[tuple[Operation, Operation]] = [
        (old_keys[k], new_keys[k]) for k in common
    ] + report.renamed_ops
    for o, n in pairs:
        for container, o_schema, n_schema in [
            ("request_body", o.body_schema, n.body_schema),
            ("response", o.response_schema, n.response_schema),
        ]:
            o_f = _fields_of(o_schema)
            n_f = _fields_of(n_schema)
            if o_f == n_f and not (o_f or n_f):
                continue
            removed = sorted(o_f - n_f)
            added = sorted(n_f - o_f)
            if not removed and not added:
                continue
            renamed = _heuristic_rename(removed, added)
            renamed_old = {a for a, _ in renamed}
            renamed_new = {b for _, b in renamed}
            fc = FieldChange(
                container=container,
                op_id=o.op_id,
                added=[x for x in added if x not in renamed_new],
                removed=[x for x in removed if x not in renamed_old],
                renamed=renamed,
            )
            if fc.added or fc.removed or fc.renamed:
                report.field_changes.append(fc)

    return report


def detect_drift(old_openapi: dict[str, Any], new_openapi: dict[str, Any]) -> DriftReport:
    return diff_operations(parse_operations(old_openapi), parse_operations(new_openapi))
