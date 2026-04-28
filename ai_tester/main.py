"""
Prototipin əsas giriş nöqtəsi.

Komandalar:
    train      RL agentini sample_backend (v1) üzərində təlim et və ssenariləri
               bilik bazasına saxla.
    drift      Cari OpenAPI sxemini izlə, əvvəlki snapshot ilə müqayisə et və
               drift hesabatı çıxar (icra olmadan).
    heal       Cari sxem ilə bilik bazasındakı ssenariləri bərpa et və icra et.
               Uğurlu/uğursuz nəticələri saxla.
    cycle      Tam Monitor-Analyze-Plan-Execute döngüsünü icra et.
    stats      Bilik bazasının statistikasını göstər.

Misal istifadə:
    # 1. Backend-i v1-də işə sal:
    SCHEMA_VERSION=v1 uvicorn sample_backend.main:app --port 8000

    # 2. Agenti təlim et və ssenariləri yığ:
    python -m ai_tester.main train --base-url http://localhost:8000 \\
        --timesteps 2000 --episodes-collect 5

    # 3. Backend-i v2-də restart et:
    SCHEMA_VERSION=v2 uvicorn sample_backend.main:app --port 8000

    # 4. Bərpa və qiymətləndirməni icra et:
    python -m ai_tester.main cycle --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ai_tester.agent import TestAgent
from ai_tester.environment import RestApiEnv
from ai_tester.knowledge_base import KnowledgeBase
from ai_tester.mape_k import MapeKController
from ai_tester.scenario_builder import extract_top_scenarios

DEFAULT_DB = "results/knowledge.db"


def cmd_train(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.db)
    env = RestApiEnv(args.base_url, episode_length=args.episode_length)
    print(f"OpenAPI yükləndi: {len(env.operations)} əməliyyat aşkarlandı.")

    # İlkin snapshot saxla
    kb.save_snapshot(env.openapi, label="train-start")

    agent = TestAgent(env, seed=args.seed)
    agent.train(total_timesteps=args.timesteps)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_path)

    # Bir neçə episode-u qiymətləndirmə rejimində işlət və tarixçələri yığ
    histories = []
    eval_results = []
    for i in range(args.episodes_collect):
        # Yeni episode üçün təzə env
        eval_env = RestApiEnv(args.base_url, episode_length=args.episode_length)
        agent.env = eval_env
        agent.vec_env.envs[0] = eval_env
        result = agent.run_episode(deterministic=False)
        eval_results.append(result)
        histories.append(eval_env.history())

    scenarios = extract_top_scenarios(histories, max_n=args.scenarios_keep)
    saved_ids = []
    for sc in scenarios:
        sid = kb.save_scenario(sc.name, source="rl", payload=sc.to_dict())
        saved_ids.append(sid)

    print("\n=== Təlim hesabatı ===")
    for i, r in enumerate(eval_results):
        print(
            f"  Episode {i}: reward={r['total_reward']:.2f} "
            f"steps={r['steps']} ops_coverage={r['ops_coverage']:.2f} "
            f"unique_2xx={r['unique_2xx_combos']}"
        )
    print(f"\nSaxlanılan ssenarilər: {len(saved_ids)} (id-lər: {saved_ids})")
    print(f"Model: {args.model_path}")


def cmd_drift(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.db)
    ctrl = MapeKController(args.base_url, kb, use_llm=not args.no_llm)
    new_snap_id, _ = ctrl.monitor(label=args.label or "drift-check")
    import sqlite3

    cx = sqlite3.connect(kb.path)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT id FROM snapshots ORDER BY id DESC LIMIT 2").fetchall()
    cx.close()
    if len(rows) < 2:
        print("Müqayisə üçün ən azı 2 snapshot lazımdır.")
        return
    new_id, old_id = int(rows[0]["id"]), int(rows[1]["id"])
    drift_id, report = ctrl.analyze(old_id, new_id)
    print(f"\n=== Drift hesabatı ({old_id} -> {new_id}) ===")
    print(report.summary())


def cmd_cycle(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.db)
    ctrl = MapeKController(args.base_url, kb, use_llm=not args.no_llm)
    cycle = ctrl.run_cycle(label=args.label or "cycle")
    if cycle is None:
        print("Heç bir dəyişiklik yoxdur, bərpa lazım deyil.")
        return
    print("\n=== MAPE-K döngüsü ===")
    print(cycle.drift_summary)
    print()
    print(f"Bərpa edilmiş ssenarilər: {cycle.repaired_count}")
    print(f"Uğurlu icra:              {cycle.successful_count}")
    print(f"Uğursuz icra:             {cycle.failed_count}")
    if cycle.repaired_count:
        rate = cycle.successful_count / cycle.repaired_count
        print(f"Bərpa müvəffəqiyyət dərəcəsi: {rate:.1%}")


def cmd_stats(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.db)
    s = kb.stats()
    print(json.dumps(s, ensure_ascii=False, indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.db)
    rows = kb.list_scenarios()
    for r in rows:
        print(
            f"  id={r['id']:3d}  source={r['source']:14s}  "
            f"parent={str(r['parent_id'] or '-'):>4s}  name={r['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_tester")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite bilik bazası yolu")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Hədəf backend-in əsas URL-i",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="LLM bərpasını söndür (yalnız rule-based)"
    )
    parser.add_argument("--label", default="", help="Snapshot/sınaq üçün izahedici label")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--timesteps", type=int, default=2000)
    p_train.add_argument("--episode-length", type=int, default=30)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--episodes-collect",
        type=int,
        default=5,
        help="Təlimdən sonra ssenari yığmaq üçün episode sayı",
    )
    p_train.add_argument(
        "--scenarios-keep",
        type=int,
        default=5,
        help="Saxlanılacaq ən yaxşı ssenari sayı",
    )
    p_train.add_argument("--model-path", default="results/ppo_agent.zip")
    p_train.set_defaults(func=cmd_train)

    p_drift = sub.add_parser("drift")
    p_drift.set_defaults(func=cmd_drift)

    p_cycle = sub.add_parser("cycle")
    p_cycle.set_defaults(func=cmd_cycle)

    p_stats = sub.add_parser("stats")
    p_stats.set_defaults(func=cmd_stats)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
