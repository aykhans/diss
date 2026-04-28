"""
Parametr dəyər mənbələri.

RL agentinin parametr dəyəri seçərkən üç mənbədən birini seçə bilməsi lazımdır:
- statik nümunələr (faker-əsaslı)
- əvvəlki cavabdan (response chaining)
- LLM (Gemini) tərəfindən sintez edilən dəyərlər

Bu modul mənbələri vahid interfeysdə təqdim edir.
"""

from __future__ import annotations

import random
import string
from typing import Any

# Sadə determine olunmuş "faker" - xarici asılılıq olmadan
_RANDOM = random.Random(42)

_NAMES = ["Buddy", "Max", "Charlie", "Lucy", "Cooper", "Bella", "Rocky", "Daisy"]
_KINDS = ["dog", "cat", "rabbit", "hamster", "parrot", "fish"]


def random_string(min_len: int = 3, max_len: int = 16) -> str:
    n = _RANDOM.randint(min_len, max_len)
    return "".join(_RANDOM.choice(string.ascii_letters) for _ in range(n))


def random_value_for_schema(schema: dict[str, Any] | None) -> Any:
    """Pydantic/JSON sxeminə uyğun təsadüfi dəyər generasiya et."""
    if not schema:
        return None
    t = schema.get("type")
    if "enum" in schema:
        return _RANDOM.choice(schema["enum"])
    if t == "string":
        # `name` və `kind` üçün xüsusi domain dəyərləri
        title = (schema.get("title") or "").lower()
        if "name" in title:
            return _RANDOM.choice(_NAMES)
        if "species" in title or "kind" in title:
            return _RANDOM.choice(_KINDS)
        min_len = schema.get("minLength", 1)
        max_len = schema.get("maxLength", 16)
        return random_string(min_len, max_len)
    if t == "integer":
        lo = schema.get("minimum", 0)
        hi = schema.get("maximum", 100)
        return _RANDOM.randint(int(lo), int(hi))
    if t == "number":
        lo = schema.get("minimum", 0)
        hi = schema.get("maximum", 100)
        return _RANDOM.uniform(lo, hi)
    if t == "boolean":
        return _RANDOM.random() > 0.5
    if t == "object":
        return {k: random_value_for_schema(v) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        n = _RANDOM.randint(0, 3)
        return [random_value_for_schema(schema.get("items")) for _ in range(n)]
    return None


class ValuePool:
    """Əvvəlki API cavablarından çıxarılmış real dəyərləri saxlayır.

    Məsələn, POST /pets cavabında qaytarılan `id` sonrakı GET /pets/{id} üçün
    real, mövcud bir dəyər kimi istifadə edilə bilər.
    """

    def __init__(self) -> None:
        self._values: dict[str, list[Any]] = {}

    def absorb(self, response_json: Any) -> None:
        """Cavabdan bütün skalyar dəyərləri toplayıb saxla."""

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        self._values.setdefault(k, []).append(v)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(response_json)

    def candidates_for(self, name: str) -> list[Any]:
        return list(self._values.get(name, []))

    def pick(self, name: str, default: Any = None) -> Any:
        cs = self.candidates_for(name)
        if not cs:
            return default
        return _RANDOM.choice(cs)
