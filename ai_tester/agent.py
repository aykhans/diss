"""
RL agent: PPO əsasında siyasət öyrənir.

Stable-Baselines3 PPO-su Dict observation space və Discrete action space ilə
işləyir. MultiInputPolicy istifadə edirik.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_tester.environment import RestApiEnv

logger = logging.getLogger(__name__)


class CoverageCallback(BaseCallback):
    """Hər episode-un sonunda coverage statistikasını loqlayır."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.coverage_history: list[dict] = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 vec_env istifadə edir; biz tək env-də işləyirik
        rewards = self.locals.get("rewards", [])
        for r in rewards:
            self._current_reward += float(r)
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.episode_rewards.append(self._current_reward)
                self._current_reward = 0.0
                # Mühitdən coverage al
                env = self.training_env.envs[0]  # type: ignore[attr-defined]
                base_env = env
                while hasattr(base_env, "env"):
                    base_env = base_env.env
                if hasattr(base_env, "coverage_summary"):
                    self.coverage_history.append(base_env.coverage_summary())
        return True


class TestAgent:
    """PPO əsaslı RL agentinin sarğısı."""

    def __init__(
        self,
        env: RestApiEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 64,
        batch_size: int = 32,
        gamma: float = 0.95,
        seed: int = 42,
    ):
        self.env = env
        # SB3 vektorlaşmış mühit gözləyir
        self.vec_env = DummyVecEnv([lambda: env])
        self.model = PPO(
            "MultiInputPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            verbose=0,
            seed=seed,
            device="cpu",
        )
        self.callback = CoverageCallback()

    def train(self, total_timesteps: int = 2000) -> None:
        logger.info("Təlim başladı: %d addım", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=False,
        )
        logger.info("Təlim bitdi.")

    def predict(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=False)
        return int(np.array(action).item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        self.model = PPO.load(str(path), env=self.vec_env, device="cpu")

    def run_episode(self, deterministic: bool = True) -> dict:
        """Bir tam episode-u test rejimində icra et."""
        obs, _info = self.env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(int(np.array(action).item()))
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        return {
            "steps": steps,
            "total_reward": total_reward,
            **self.env.coverage_summary(),
        }
