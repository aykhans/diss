"""
Hyperparameter həssaslıq analizi.

PPO agentinin endpoint əhatə dairəsinin iki əsas hiperparametrdən asılılığını
ölçür:
  - timesteps: təlim üçün ümumi addım sayı
  - episode_length: hər episode-dakı addımların sayı

Eyni seed (42) ilə bütün konfiqurasiyalar eyni başlanğıc nöqtəsindən təlim
alır, beləliklə nəticələr birbaşa müqayisə edilə bilir.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_tester.agent import TestAgent
from ai_tester.environment import RestApiEnv

BASE_URL = "http://localhost:8000"


def run_one(timesteps: int, episode_length: int, seed: int = 42, n_eval_episodes: int = 5) -> dict:
    env = RestApiEnv(BASE_URL, episode_length=episode_length)
    t0 = time.time()
    agent = TestAgent(env, seed=seed)
    agent.train(total_timesteps=timesteps)
    train_time = time.time() - t0

    coverages = []
    rewards = []
    unique_combos = []
    for _ in range(n_eval_episodes):
        eval_env = RestApiEnv(BASE_URL, episode_length=episode_length)
        agent.env = eval_env
        agent.vec_env.envs[0] = eval_env
        result = agent.run_episode(deterministic=False)
        coverages.append(result["ops_coverage"])
        rewards.append(result["total_reward"])
        unique_combos.append(result["unique_2xx_combos"])
    return {
        "timesteps": timesteps,
        "episode_length": episode_length,
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "mean_coverage": round(sum(coverages) / len(coverages), 3),
        "max_coverage": round(max(coverages), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 2),
        "max_unique_combos": max(unique_combos),
    }


def main() -> None:
    out_path = Path("results/hyperparam_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Timesteps sweep (episode_length sabit = 25)
    timestep_results = []
    for ts in [500, 1500, 5000]:
        print(f"\n>>> timesteps={ts}, episode_length=25 ...")
        r = run_one(ts, 25)
        print(
            f"    coverage={r['mean_coverage']:.2f}, "
            f"unique_combos={r['max_unique_combos']}, "
            f"time={r['train_time_s']}s"
        )
        timestep_results.append(r)

    # 2. Episode length sweep (timesteps sabit = 1500)
    length_results = []
    for el in [15, 25, 40]:
        print(f"\n>>> timesteps=1500, episode_length={el} ...")
        r = run_one(1500, el)
        print(
            f"    coverage={r['mean_coverage']:.2f}, "
            f"unique_combos={r['max_unique_combos']}, "
            f"time={r['train_time_s']}s"
        )
        length_results.append(r)

    payload = {
        "timesteps_sweep": timestep_results,
        "episode_length_sweep": length_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
