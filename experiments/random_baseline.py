"""
Random baseline: təlim olmadan təsadüfi seçim ilə endpoint əhatə dairəsi.

PPO agentinin əldə etdiyi əhatə dairəsinin nə qədər "təlimdən" gəldiyini
yoxlamaq üçün təsadüfi seçim baseline-ı ilə müqayisə aparılır. Eyni episode
uzunluğu (25) və eyni episode sayı (5) istifadə olunur.

H1 hipotezi: PPO təsadüfi seçimdən əhəmiyyətli dərəcədə yüksək əhatə təmin edir.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_tester.environment import RestApiEnv

BASE_URL = "http://localhost:8000"


def run_random_episode(env: RestApiEnv, rng: random.Random) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action = rng.randint(0, len(env.operations) - 1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return {
        "total_reward": total_reward,
        **env.coverage_summary(),
    }


def run_random_baseline(seed: int, n_episodes: int = 5, episode_length: int = 25) -> dict:
    rng = random.Random(seed)
    coverages = []
    rewards = []
    unique_combos = []
    for _ in range(n_episodes):
        env = RestApiEnv(BASE_URL, episode_length=episode_length)
        result = run_random_episode(env, rng)
        coverages.append(result["ops_coverage"])
        rewards.append(result["total_reward"])
        unique_combos.append(result["unique_2xx_combos"])
    return {
        "seed": seed,
        "n_episodes": n_episodes,
        "episode_length": episode_length,
        "mean_coverage": round(sum(coverages) / len(coverages), 3),
        "max_coverage": round(max(coverages), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 2),
        "max_unique_combos": max(unique_combos),
        "per_episode_coverages": coverages,
        "per_episode_rewards": rewards,
    }


def main() -> None:
    results = []
    for seed in [42, 123, 7]:
        print(f"\n>>> Random baseline, seed={seed}")
        r = run_random_baseline(seed)
        print(
            f"    coverage={r['mean_coverage']:.2f}, "
            f"unique_combos={r['max_unique_combos']}, "
            f"reward={r['mean_reward']:.2f}"
        )
        results.append(r)

    out = Path("results/random_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seeds": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
