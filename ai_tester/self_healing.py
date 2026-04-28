"""
Self-healing modulu: Gemini API əsasında.

Konsept dəyişməsi detektoru hesabat verdikdən sonra bu modul:
1. Köhnə test ssenarisini (uğurlu kombinasiyaları) götürür
2. Drift hesabatını və yeni OpenAPI-ni Gemini-yə kontekst kimi verir
3. Modeldən yenilənmiş test ssenarisi sintez etməsini istəyir
4. Sintez edilmiş ssenarini real backend-də sınaqdan keçirir
5. Uğurlu olarsa təsdiq edir, olmazsa rule-based fallback işlədir.

Gemini API açarı `GEMINI_API_KEY` mühit dəyişənində olmalıdır. Açar yoxdursa
modul yalnız rule-based bərpaya keçir (heuristic field renaming, path suffix
matching).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_tester.concept_drift import DriftReport
from ai_tester.openapi_loader import Operation

logger = logging.getLogger(__name__)


@dataclass
class TestStep:
    method: str
    path: str
    path_args: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestScenario:
    """Bir əməliyyat ardıcıllığını ifadə edən test ssenarisi."""

    name: str
    steps: list[TestStep] = field(default_factory=list)
    expected_status_codes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [asdict(s) for s in self.steps],
            "expected_status_codes": self.expected_status_codes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestScenario":
        return cls(
            name=d["name"],
            steps=[TestStep(**s) for s in d["steps"]],
            expected_status_codes=d.get("expected_status_codes", []),
        )


# ---------------------- Rule-based bərpa ----------------------


def _required_fields(schema: dict[str, Any] | None) -> list[str]:
    if not schema or not isinstance(schema, dict):
        return []
    return list(schema.get("required", []))


def _properties(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not schema or not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    return props


def _default_for_type(prop: dict[str, Any]) -> Any:
    """Pydantic/JSON sxem tipinə əsasən sadə default dəyər qaytar."""
    t = prop.get("type")
    if t == "string":
        return "x"
    if t == "integer":
        lo = prop.get("minimum", 1)
        return int(lo)
    if t == "number":
        return float(prop.get("minimum", 1.0))
    if t == "boolean":
        return True
    if t == "array":
        return []
    if t == "object":
        return {}
    return None


def _adapt_payload(
    old_payload: dict[str, Any],
    new_schema: dict[str, Any] | None,
    explicit_renames: dict[str, str],
) -> dict[str, Any]:
    """Köhnə payload-ı yeni sxemə uyğunlaşdır.

    1. Açıq rename-ləri tətbiq et
    2. Yeni sxemdə yer almayan sahələri at
    3. Çatışmayan required sahələr üçün default dəyər qoy
    4. Heuristik: ad oxşarlığına və ya tipə əsasən "yetim" sahələri yenilərinə map et
    """
    new_props = _properties(new_schema)
    if not new_props:
        return dict(old_payload)

    # 1. açıq rename
    payload: dict[str, Any] = {}
    for k, v in old_payload.items():
        nk = explicit_renames.get(k, k)
        payload[nk] = v

    # 2. yeni sxemdə yer alanları saxla, "yetimləri" başqa yerə map etməyə çalış
    kept: dict[str, Any] = {}
    orphans: dict[str, Any] = {}
    for k, v in payload.items():
        if k in new_props:
            kept[k] = v
        else:
            orphans[k] = v

    # 3. heuristik uyğunlaşdırma: yetim sahələri çatışmayan yeni sahələrə map et
    missing = [k for k in new_props if k not in kept]
    for orph_name, orph_val in list(orphans.items()):
        # ad oxşarlığı (alt-sətir uyğunluğu)
        match = None
        for m in missing:
            if orph_name in m or m in orph_name:
                match = m
                break
        # tip uyğunluğu
        if match is None:
            for m in missing:
                want_t = new_props[m].get("type")
                if want_t == "string" and isinstance(orph_val, str):
                    match = m
                    break
                if (
                    want_t == "integer"
                    and isinstance(orph_val, int)
                    and not isinstance(orph_val, bool)
                ):
                    match = m
                    break
                if want_t == "number" and isinstance(orph_val, (int, float)):
                    match = m
                    break
        if match is not None:
            kept[match] = orph_val
            missing.remove(match)
            del orphans[orph_name]

    # 4. çatışan required sahələr üçün default
    for req in _required_fields(new_schema):
        if req not in kept:
            kept[req] = _default_for_type(new_props.get(req, {}))

    return kept


def rule_based_repair(
    scenario: TestScenario,
    drift: DriftReport,
    new_ops: list[Operation],
) -> TestScenario:
    """LLM olmadan, drift hesabatına və evristikalara əsaslanan bərpa."""
    new_index: dict[tuple[str, str], Operation] = {(o.method, o.path): o for o in new_ops}
    rename_map = {old.path: new.path for old, new in drift.renamed_ops}

    # Hər əməliyyat üçün açıq rename-lər
    explicit_renames: dict[str, dict[str, str]] = {}
    for fc in drift.field_changes:
        if fc.container == "request_body" and fc.renamed:
            explicit_renames.setdefault(fc.op_id, {}).update({a: b for a, b in fc.renamed})

    repaired_steps: list[TestStep] = []
    for step in scenario.steps:
        path = step.path
        # path rename (renamed_ops siyahısına əsasən)
        if path in rename_map:
            path = rename_map[path]
        # path hələ də yoxdursa, son seqment uyğunluğu ilə axtar
        if (step.method, path) not in new_index:
            tail = path.split("/")[-1]
            for op in new_ops:
                if op.method == step.method and op.path.split("/")[-1] == tail:
                    path = op.path
                    break

        # uyğun yeni əməliyyat
        target = new_index.get((step.method, path))
        new_schema = target.body_schema if target else None
        # əməliyyat üçün rename-ləri yığ; eyni op_id olmayadıqda qlobal rename-ləri də əlavə et
        renames_for_op = dict(explicit_renames.get(target.op_id, {})) if target else {}
        if not renames_for_op:
            for r in explicit_renames.values():
                for k, v in r.items():
                    renames_for_op.setdefault(k, v)

        new_payload = _adapt_payload(step.payload, new_schema, renames_for_op)

        repaired_steps.append(
            TestStep(
                method=step.method,
                path=path,
                path_args=dict(step.path_args),
                payload=new_payload,
            )
        )
    return TestScenario(
        name=scenario.name + " (rule-based repaired)",
        steps=repaired_steps,
        expected_status_codes=scenario.expected_status_codes,
    )


# ---------------------- Gemini-əsaslı bərpa ----------------------

_GEMINI_PROMPT = """Sən REST API testlərinin self-healing eksperti rolundasan. Test ssenarisi köhnə \
API sxemi üçün yazılıb, lakin API təkamül edib və ssenari pozulub.

KÖHNƏ TEST SSENARİSİ (JSON):
{old_scenario}

API DƏYİŞİKLİYİ HESABATI:
{drift_summary}

YENİ API ƏMƏLIYYATLARI (method + path siyahısı):
{new_ops}

YENİ API REQUEST/RESPONSE SXEMLƏRİ (qısaldılmış JSON):
{new_schemas}

Vəzifə: Yuxarıdakı dəyişiklikləri nəzərə alaraq köhnə ssenarini yenilə. Yeni \
ssenari real fəaliyyət göstərə bilməlidir. Yalnız aşağıdakı dəqiq JSON formatında \
cavab ver, başqa heç bir mətn olmamalıdır:

{{
  "name": "string",
  "steps": [
    {{"method": "GET|POST|PUT|DELETE", "path": "/...", "path_args": {{}}, "payload": {{}}}}
  ],
  "expected_status_codes": [200, 201, ...]
}}
"""


def _try_import_gemini():
    try:
        import google.generativeai as genai  # type: ignore

        return genai
    except Exception as e:  # pragma: no cover
        logger.info("google-generativeai mövcud deyil: %s", e)
        return None


def gemini_repair(
    scenario: TestScenario,
    drift: DriftReport,
    new_ops: list[Operation],
    model_name: str = "gemini-2.5-flash",
) -> TestScenario | None:
    """Gemini API ilə test ssenarisinin bərpası. Açar yoxdursa None qaytarır."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY təyin edilməyib; LLM bərpası ötürülür.")
        return None

    genai = _try_import_gemini()
    if genai is None:
        return None

    genai.configure(api_key=api_key)
    new_ops_text = "\n".join(o.display() for o in new_ops)
    new_schemas = {
        o.display(): {
            "request": _shrink_schema(o.body_schema),
            "response": _shrink_schema(o.response_schema),
        }
        for o in new_ops
    }
    prompt = _GEMINI_PROMPT.format(
        old_scenario=json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2),
        drift_summary=drift.summary(),
        new_ops=new_ops_text,
        new_schemas=json.dumps(new_schemas, ensure_ascii=False, indent=2)[:4000],
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Bəzən LLM ```json ... ``` blokunu qaytarır
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        return TestScenario.from_dict(data)
    except Exception as exc:  # pragma: no cover - şəbəkə/parse səhvi
        logger.warning("Gemini bərpası uğursuz oldu: %s", exc)
        return None


def _shrink_schema(s: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sxeminin JSON-da çox yer tutmaması üçün yalnız sahə adlarını və tiplərini saxla."""
    if not s or not isinstance(s, dict):
        return s
    if s.get("type") == "object" and "properties" in s:
        return {
            "type": "object",
            "properties": {
                k: {"type": v.get("type"), "format": v.get("format")}
                for k, v in s["properties"].items()
                if isinstance(v, dict)
            },
            "required": s.get("required", []),
        }
    return {"type": s.get("type")}


# ---------------------- Yüksək səviyyəli API ----------------------


def heal_scenario(
    scenario: TestScenario,
    drift: DriftReport,
    new_ops: list[Operation],
    use_llm: bool = True,
) -> tuple[TestScenario, str]:
    """Ssenarini bərpa et. Mənbə (`llm` və ya `rule`) ilə birlikdə qaytar."""
    if use_llm:
        repaired = gemini_repair(scenario, drift, new_ops)
        if repaired is not None:
            return repaired, "llm"
    return rule_based_repair(scenario, drift, new_ops), "rule"
