# lunarlander_dlo.py
# D-LO (Dual Local/Global Observation) + PPO for LunarLander-v3
# - Reward shaping for stable, centered, gentle landings
# - PPO hyperparams tuned for reliable learning
# - Zoom-on-danger HUD during eval
#
# Usage:
#   python lunarlander_dlo.py train --timesteps 200000 --model ppo_lander_dlo.zip
#   python lunarlander_dlo.py eval  --model ppo_lander_dlo.zip

import argparse
import warnings
import time
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Quiet down some gym warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)


# =========================
# D-LO Wrapper for LunarLander
# =========================
class LanderDLOWrapper(gym.Wrapper):
    """
    Wraps LunarLander-v3:
    - Fuses a simple multi-scale feature: [0.5*obs, 1.0*obs, 2.0*obs]
      so policy can attend to both coarse (global) and amplified (local) signals.
    - Adds reward shaping that strongly encourages stable, centered, soft landing.
    - Provides a zoom-on-danger render for demos.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # The base observation is an 8D float vector
        assert isinstance(env.observation_space, spaces.Box)
        base_low = env.observation_space.low.astype(np.float32)
        base_high = env.observation_space.high.astype(np.float32)

        # Our fused observation is 3x the length: small/med/large scales
        low = np.concatenate([0.5 * base_low, base_low, 2.0 * base_low]).astype(np.float32)
        high = np.concatenate([0.5 * base_high, base_high, 2.0 * base_high]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internals for reward shaping / zoom logic
        self._last_dist = None
        self._zoom_state = "global"  # "global" or "local"
        self._danger = False

    # ---------- D-LO observation ----------
    def _fuse_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        return np.concatenate([0.5 * obs, obs, 2.0 * obs]).astype(np.float32)

    # ---------- Distance to target pad (x,y ~ 0,0 is center) ----------
    @staticmethod
    def _dist_to_pad(obs: np.ndarray) -> float:
        x, y = float(obs[0]), float(obs[1])
        return np.sqrt(x * x + y * y)

    # ---------- Reward shaping ----------
    def _shape_reward(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        # Decompose original obs for clarity
        x, y, vx, vy, theta, vtheta, leg_l, leg_r = [float(v) for v in obs]

        shaped = 0.0

        # Encourage being centered over pad
        dist = self._dist_to_pad(obs)
        if self._last_dist is not None:
            # Reward approaching the pad (positive if getting closer)
            shaped += 0.5 * (self._last_dist - dist)
        self._last_dist = dist

        # Penalize tilt and angular velocity
        shaped += -0.10 * abs(theta)       # keep upright
        shaped += -0.02 * abs(vtheta)      # reduce spin

        # Penalize vertical/horizontal speed (encourage gentle descent)
        shaped += -0.04 * abs(vy)
        shaped += -0.02 * abs(vx)

        # Small living penalty (discourage dithering high in the air)
        shaped += -0.005

        # Bonus for leg contact (stable landing posture)
        if leg_l > 0.5:
            shaped += 0.3
        if leg_r > 0.5:
            shaped += 0.3

        # Terminal shaping: large bonus for good landing, penalty for crash
        if terminated:
            # Heuristic: if final env reward is large, it's likely a successful landing
            if reward > 100:     # typical landing reward > 100
                shaped += 60.0   # success bonus
            else:
                shaped += -60.0  # crash penalty

        # Combine
        return reward + shaped

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_dist = self._dist_to_pad(obs)
        self._zoom_state = "global"
        self._danger = False
        return self._fuse_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reward shaping
        reward = self._shape_reward(obs, reward, terminated, truncated, info)

        # Update zoom state based on danger: high speed or tilt or far from pad
        x, y, vx, vy, theta, vtheta, leg_l, leg_r = [float(v) for v in obs]
        speed = np.sqrt(vx * vx + vy * vy)
        self._danger = (abs(theta) > 0.2) or (speed > 0.6) or (self._dist_to_pad(obs) > 0.5)
        self._zoom_state = "local" if self._danger else "global"

        return self._fuse_obs(obs), reward, terminated, truncated, info

    # ---------- Zoom-on-danger render for demo ----------
    def render(self, mode: Optional[str] = None):
        # Always ask underlying env for rgb frame (works with render_mode="rgb_array")
        frame = self.env.render()
        if frame is None:
            return None

        frame = np.array(frame)
        h, w = frame.shape[:2]

        # Zoom if in danger, otherwise show a mild framed view
        if self._zoom_state == "local":
            # tighter crop when danger
            zoom_factor = 1.35
            cx, cy = w // 2, h // 2
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
            x2, y2 = min(w, cx + nw // 2), min(h, cy + nh // 2)
            cropped = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (w, h))
            cv2.putText(frame, "ZOOM (danger)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 255), 2)
        else:
            # slight letterbox & green frame when safe
            pad = 8
            cv2.rectangle(frame, (pad, pad), (w - pad, h - pad), (0, 180, 0), 3)

        return frame


# =========================
# Feature Extractor (for PPO MLP policy)
# =========================
class DLOFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple MLP over fused 24-D observation (8 * 3 scales).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float().view(observations.shape[0], -1)
        return self.net(x)


# =========================
# Env factory
# =========================
def make_env(render: bool = False, seed: Optional[int] = None):
    render_mode = "rgb_array" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    env = LanderDLOWrapper(env)
    return env


# =========================
# Training
# =========================
def train(total_timesteps: int = 200_000, save_path: str = "ppo_lander_dlo.zip", seed: int = 42):
    set_random_seed(seed)
    env = DummyVecEnv([lambda: make_env(render=False, seed=seed)])

    policy_kwargs = dict(
        features_extractor_class=DLOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],  # actor/critic MLP
    )

    # Tuned for stable landing
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        device="cpu",
    )

    print("🚀 Training PPO (LunarLander-v3 + D-LO + reward shaping)…")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")


# =========================
# Evaluation
# =========================
def evaluate(model_path: str = "ppo_lander_dlo.zip", episodes: int = 5, seed: int = 123):
    env = DummyVecEnv([lambda: make_env(render=True, seed=seed)])
    model = PPO.load(model_path)

    print("🎮 AI Agent playing…  (Press 'q' to stop)\n")
    try:
        for ep in range(1, episodes + 1):
            obs = env.reset()
            done = False
            ep_ret = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_ret += float(reward)

                # Get frame and overlay return
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frame = np.array(frame)
                    cv2.putText(frame, f"Episode: {ep} | Return: {ep_ret:.1f}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    cv2.imshow("LunarLander D-LO", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopping (Q pressed).")
                        raise KeyboardInterrupt

            print(f"🏁 Episode {ep}: Return = {ep_ret:.2f}")
            time.sleep(0.75)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        time.sleep(0.3)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--model", type=str, default="ppo_lander_dlo.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, save_path=args.model, seed=args.seed)
    else:
        evaluate(model_path=args.model, episodes=5, seed=args.seed + 1)
