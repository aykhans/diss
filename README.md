# AI-əsaslı Adaptiv Test Prototipi

Magistr dissertasiyasının III fəsli üçün hazırlanmış prototip: backend
(REST API) sistemləri üçün möhkəmləndirici öyrənmə (RL) və böyük dil modeli
(LLM) kombinasiyası ilə **adaptiv** test ssenariləri yaradır və API
təkamülündən sonra onları **avtomatik bərpa edir** (self-healing).

## Komponentlər

```
proyekt/
├── sample_backend/     FastAPI Pet Store (v1 və v2 sxemləri)
└── ai_tester/
    ├── openapi_loader.py   OpenAPI sxemini yüklə və əməliyyatları parse et
    ├── value_pool.py       Parametr dəyər mənbələri (random + chained)
    ├── environment.py      RL mühiti (Gymnasium)
    ├── agent.py            PPO agent (Stable-Baselines3)
    ├── concept_drift.py    OpenAPI fərq detektoru
    ├── self_healing.py     Gemini + rule-based test bərpası
    ├── knowledge_base.py   SQLite bilik bazası
    ├── mape_k.py           MAPE-K orkestrator
    ├── scenario_builder.py Episode tarixçəsindən ssenari çıxarma
    └── main.py             CLI giriş nöqtəsi
```

## Quraşdırma (uv ilə)

```bash
cd proyekt
uv sync                          # əsas asılılıqlar
uv sync --extra experiments      # eksperiment skriptləri üçün matplotlib
uv sync --group dev              # ruff və başqa inkişaf alətləri
```

`uv sync` avtomatik olaraq `.venv/` yaradır və `uv.lock` faylına uyğun
versiyaları qurur. Aktivləşdirmə tələb olunmur, bütün əmrləri `uv run`
prefiksi ilə çağırın.

## Tipik iş axını

### 1. Backend-i v1 sxemində işə sal
```bash
SCHEMA_VERSION=v1 uv run uvicorn sample_backend.main:app --port 8000
```

### 2. RL agentini təlim et və ssenariləri topla
```bash
uv run ai-tester train --timesteps 3000 --episodes-collect 5
```

### 3. Backend-i v2 sxeminə keç (sxem dəyişikliyi simulyasiyası)
```bash
SCHEMA_VERSION=v2 uv run uvicorn sample_backend.main:app --port 8000
```

### 4. MAPE-K döngüsünü icra et (drift aşkarlanmalı, ssenarilər bərpa edilməli)
```bash
uv run ai-tester cycle
```

### 5. Statistikanı oxu
```bash
uv run ai-tester stats
uv run ai-tester list
```

### Eksperimentlər (matplotlib lazımdır)
```bash
uv run --extra experiments python experiments/run_evaluation.py
uv run --extra experiments python experiments/hyperparameter_sweep.py
uv run --extra experiments python experiments/random_baseline.py
```

### Kod keyfiyyəti
```bash
uv run ruff check .
uv run ruff format .
```

## Konfiqurasiya

| Mühit dəyişəni     | Təyinat                                                         |
|--------------------|------------------------------------------------------------------|
| `GEMINI_API_KEY`   | Gemini-əsaslı self-healing üçün API açarı (yoxdursa rule-based) |
| `SCHEMA_VERSION`   | sample backend üçün `v1` və ya `v2`                             |

## Dissertasiya ilə əlaqə

Bu prototip 2.4.9 yarımfəslində təklif edilmiş arxitekturanın reallaşdırılmasıdır:

| 2.4.9 komponenti                     | Reallaşdırma                       |
|--------------------------------------|------------------------------------|
| Möhkəmləndirici öyrənmə nüvəsi       | `agent.py` (PPO, Stable-Baselines3) |
| Self-healing modulu                  | `self_healing.py` (Gemini + rule)   |
| Konsept dəyişməsi detektoru          | `concept_drift.py`                  |
| Bilik bazası                         | `knowledge_base.py` (SQLite)        |
| MAPE-K döngüsü                       | `mape_k.py`                         |
