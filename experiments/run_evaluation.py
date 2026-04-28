"""
Tam qiymətləndirmə skripti: dissertasiyanın 3.3 yarımfəslinin verilənlərini
istehsal edir.

Eksperimentin mərhələləri:
    1. v1 backend-i işə salınır (xarici ssh).
    2. RL agent N müxtəlif seed üzrə təlim olunur (3 seed).
    3. Hər seed üçün test ssenariləri toplanır.
    4. v2 backend-ə keçirilir (avtomatik portu yenidən yükləyirik).
    5. Konsept dəyişməsi aşkarlanır.
    6. Hər seedin ssenariləri rule-based və (mümkündürsə) LLM-əsaslı bərpa
       ilə təmir edilir, real backend üzərində icra olunur.
    7. Nəticələr (coverage, repair success, repair source) JSON-a yazılır.

Bu skript backend-i özü idarə etmir: istifadəçi uvicorn-u manual başlatmalı,
v1 ilə başlanmalı, agentlər təlim olunduqdan sonra v2-yə keçirməlidir. Hər
mərhələ üçün skript dayanır və istifadəçidən təsdiq tələb edir (dialoq
rejimi). Avtomatik rejim üçün `--auto` bayrağı.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_tester.environment import RestApiEnv
from ai_tester.agent import TestAgent
from ai_tester.scenario_builder import extract_top_scenarios
from ai_tester.openapi_loader import fetch_openapi, parse_operations
from ai_tester.concept_drift import detect_drift
from ai_tester.self_healing import (
    TestScenario,
    rule_based_repair,
    gemini_repair,
)


def wait_for_health(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2.0)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def run_episode_collection(
    base_url: str, seed: int, timesteps: int, episodes: int, episode_length: int
) -> dict:
    """Bir seed üçün təlim et, ssenariləri yığ və coverage-i ölç."""
    env = RestApiEnv(base_url, episode_length=episode_length)
    n_ops = len(env.operations)
    agent = TestAgent(env, seed=seed)
    agent.train(total_timesteps=timesteps)

    histories = []
    rewards = []
    coverages = []
    unique_combos = []
    for _ in range(episodes):
        eval_env = RestApiEnv(base_url, episode_length=episode_length)
        agent.env = eval_env
        agent.vec_env.envs[0] = eval_env
        result = agent.run_episode(deterministic=False)
        rewards.append(result["total_reward"])
        coverages.append(result["ops_coverage"])
        unique_combos.append(result["unique_2xx_combos"])
        histories.append(eval_env.history())

    scenarios = extract_top_scenarios(histories, max_n=5)
    return {
        "seed": seed,
        "n_operations": n_ops,
        "episode_rewards": rewards,
        "episode_coverages": coverages,
        "unique_2xx_combos": unique_combos,
        "scenarios": [sc.to_dict() for sc in scenarios],
        "openapi": env.openapi,
    }


def run_repair_phase(base_url: str, train_data: list[dict], use_llm: bool) -> list[dict]:
    """Yeni backend (v2) üzərində bərpa edib qiymətləndir."""
    new_openapi = fetch_openapi(base_url)
    new_ops = parse_operations(new_openapi)

    results = []
    for td in train_data:
        old_openapi = td["openapi"]
        report = detect_drift(old_openapi, new_openapi)

        per_seed = {
            "seed": td["seed"],
            "drift": {
                "renamed_ops": [(o.display(), n.display()) for o, n in report.renamed_ops],
                "n_renamed_ops": len(report.renamed_ops),
                "n_field_changes": len(report.field_changes),
                "n_new_ops": len(report.new_ops),
                "n_removed_ops": len(report.removed_ops),
            },
            "scenarios": [],
        }

        for sc_dict in td["scenarios"]:
            scenario = TestScenario.from_dict(sc_dict)
            # 1. Rule-based bərpa
            repaired_rule = rule_based_repair(scenario, report, new_ops)
            rule_codes = _execute(base_url, repaired_rule)
            rule_success = _is_repair_successful(rule_codes)

            entry = {
                "name": scenario.name,
                "n_steps": len(scenario.steps),
                "rule_status_codes": rule_codes,
                "rule_success": rule_success,
            }

            # 2. LLM-əsaslı bərpa (yalnız açar varsa)
            if use_llm and os.getenv("GEMINI_API_KEY"):
                llm_scenario = gemini_repair(scenario, report, new_ops)
                if llm_scenario is not None:
                    llm_codes = _execute(base_url, llm_scenario)
                    entry["llm_status_codes"] = llm_codes
                    entry["llm_success"] = _is_repair_successful(llm_codes)
                else:
                    entry["llm_success"] = None
            per_seed["scenarios"].append(entry)
        results.append(per_seed)
    return results


def _execute(base_url: str, scenario: TestScenario) -> list[int]:
    sess = requests.Session()
    codes = []
    for step in scenario.steps:
        url = base_url + step.path
        for k, v in step.path_args.items():
            url = url.replace("{" + k + "}", str(v))
        try:
            r = sess.request(
                step.method,
                url,
                json=step.payload if step.payload else None,
                timeout=4.0,
            )
            codes.append(r.status_code)
        except requests.RequestException:
            codes.append(599)
    return codes


def _is_repair_successful(codes: list[int]) -> bool:
    """Sxem-bərpasına dair uğur: 422/5xx yox, ən azı bir 2xx."""
    if not codes:
        return False
    no_schema_err = all(c not in (422,) and c < 500 for c in codes)
    any_2xx = any(200 <= c < 300 for c in codes)
    return no_schema_err and any_2xx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--timesteps", type=int, default=1500)
    p.add_argument("--episode-length", type=int, default=25)
    p.add_argument("--episodes-collect", type=int, default=5)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out", default="results/evaluation.json")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--phase", choices=["train", "repair", "both"], default="both")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.phase in ("train", "both"):
        if not wait_for_health(args.base_url):
            print(f"ERROR: backend cavab vermir: {args.base_url}", file=sys.stderr)
            return 1
        v_initial = requests.get(f"{args.base_url}/health").json().get("version")
        print(f"İlkin backend versiyası: {v_initial}")

        train_data = []
        for seed in args.seeds:
            print(f"\n=== Seed {seed} təlimi ===")
            data = run_episode_collection(
                base_url=args.base_url,
                seed=seed,
                timesteps=args.timesteps,
                episodes=args.episodes_collect,
                episode_length=args.episode_length,
            )
            print(
                f"  ops_coverage avg: "
                f"{sum(data['episode_coverages']) / len(data['episode_coverages']):.2f}"
            )
            train_data.append(data)

        train_out = out.with_name(out.stem + ".train.json")
        with open(train_out, "w") as f:
            json.dump(
                {
                    "initial_version": v_initial,
                    "seeds": train_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nTrain nəticələri saxlanıldı: {train_out}")

        if args.phase == "train":
            return 0

    if args.phase in ("repair", "both"):
        train_path = out.with_name(out.stem + ".train.json")
        if not train_path.exists():
            print(
                f"ERROR: train fazasının çıxışı tapılmadı: {train_path}",
                file=sys.stderr,
            )
            return 1

        print("\n>>> İndi backend-i v2-yə keçirin (yeni terminalda):")
        print("    pkill -f 'uvicorn sample_backend' && \\")
        print("    SCHEMA_VERSION=v2 .venv/bin/uvicorn sample_backend.main:app --port 8000")
        try:
            input("Hazır olduqda Enter basın... ")
        except EOFError:
            pass

        if not wait_for_health(args.base_url):
            print(f"ERROR: backend cavab vermir: {args.base_url}", file=sys.stderr)
            return 1
        v_after = requests.get(f"{args.base_url}/health").json().get("version")
        print(f"Yeni backend versiyası: {v_after}")

        with open(train_path) as f:
            train_payload = json.load(f)
        repair_results = run_repair_phase(
            base_url=args.base_url,
            train_data=train_payload["seeds"],
            use_llm=args.use_llm,
        )

        # Cəmi statistika
        rule_success_total = 0
        rule_total = 0
        llm_success_total = 0
        llm_total = 0
        for r in repair_results:
            for sc in r["scenarios"]:
                rule_total += 1
                if sc.get("rule_success"):
                    rule_success_total += 1
                if "llm_success" in sc and sc["llm_success"] is not None:
                    llm_total += 1
                    if sc["llm_success"]:
                        llm_success_total += 1
        summary = {
            "from_version": train_payload["initial_version"],
            "to_version": v_after,
            "n_seeds": len(repair_results),
            "rule_success_rate": (rule_success_total / rule_total if rule_total else 0.0),
            "llm_success_rate": (llm_success_total / llm_total if llm_total else None),
            "rule_total": rule_total,
            "llm_total": llm_total,
        }
        full = {"summary": summary, "per_seed": repair_results}
        with open(out, "w") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)

        print("\n=== Yekun ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Tam nəticələr: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
