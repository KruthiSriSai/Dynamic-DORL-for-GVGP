# hillclimb_dlo.py
# Dual-Observation (Global + Local) demo on MountainCarContinuous-v0
# - D-LO wrapper fuses multi-scale numeric observations
# - Visual zoom-in near steep slopes / high speed / near-goal, zoom-out otherwise
# - Reward shaping to learn clean momentum-and-climb behavior
# - PPO + SB3; Q to quit during eval; smooth, readable HUD

import argparse
import math
import time

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# =========================
# D-LO WRAPPER
# =========================
class DLOWrapper(gym.Wrapper):
    """
    For MountainCarContinuous-v0 (obs = [position, velocity])
    - D-LO numeric fusion: [obs*0.5, obs, obs*2.0] -> 6D vector
    - Reward shaping for stable, impressive climbing
    - Zoom logic:
        local-zoom when |slope| high OR |vel| high OR near goal
        otherwise global
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        base = env.observation_space  # Box(low=[-1.2, -0.07], high=[0.6, 0.07])
        assert isinstance(base, spaces.Box) and base.shape == (2,)

        # Expanded observation: 3x
        low = np.concatenate([base.low * 0.5, base.low, base.low * 2.0]).astype(np.float32)
        high = np.concatenate([base.high * 0.5, base.high, base.high * 2.0]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.zoom_state = "global"  # or "local"
        self._last_obs = None

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        # obs is 2D: [pos, vel]
        small = obs * 0.5
        med = obs
        large = obs * 2.0
        fused = np.concatenate([small, med, large], axis=0).astype(np.float32)
        return fused

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.zoom_state = "global"
        self._last_obs = obs
        return self._encode(obs), info

    def step(self, action):
        # Step base env
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        pos, vel = float(obs[0]), float(obs[1])

        # Reward shaping:
        # - keep original reward (>= 100 when goal reached)
        shaped = base_reward
        # - encourage moving right (toward goal)
        shaped += 0.5 * pos
        # - reward healthy speed in either direction (for momentum building)
        shaped += 0.1 * abs(vel)
        # - penalize large throttle (to look smoother)
        if isinstance(action, (list, tuple, np.ndarray)):
            throttle = float(np.clip(action[0], -1.0, 1.0))
        else:
            throttle = float(np.clip(action, -1.0, 1.0))
        shaped -= 0.05 * (throttle ** 2)
        # - small survival bonus
        shaped += 0.01

        # Update zoom logic:
        slope = 3.0 * math.cos(3.0 * pos)  # derivative of sin(3x)
        near_goal = (pos > 0.45)  # goal ~0.45..0.6
        if abs(slope) > 1.5 or abs(vel) > 0.035 or near_goal:
            self.zoom_state = "local"
        else:
            self.zoom_state = "global"

        self._last_obs = obs
        return self._encode(obs), shaped, terminated, truncated, info

    def render(self):
        # We assume base env created with render_mode="rgb_array" during eval
        frame = self.env.render()
        if frame is None:
            return None

        # Zoom effect
        h, w, _ = frame.shape
        if self.zoom_state == "local":
            z = 1.35
            cx, cy = w // 2, h // 2
            nw, nh = int(w / z), int(h / z)
            x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
            x2, y2 = min(w, cx + nw // 2), min(h, cy + nh // 2)
            crop = frame[y1:y2, x1:x2]
            frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            z = 0.95
            small = cv2.resize(frame, (int(w * z), int(h * z)))
            boxed = np.zeros_like(frame)
            y_off = (h - small.shape[0]) // 2
            x_off = (w - small.shape[1]) // 2
            boxed[y_off:y_off + small.shape[0], x_off:x_off + small.shape[1]] = small
            frame = boxed

        # Draw simple HUD
        if self._last_obs is not None:
            pos, vel = float(self._last_obs[0]), float(self._last_obs[1])
            hud = f"Pos: {pos:+.3f}  Vel: {vel:+.3f}  View: {'LOCAL' if self.zoom_state=='local' else 'GLOBAL'}"
            cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)

        # Subtle border color by mode
        color = (60, 210, 80) if self.zoom_state == "global" else (40, 90, 240)
        cv2.rectangle(frame, (8, 8), (w - 8, h - 8), color, 3)

        return frame

# =========================
# Feature Extractor (MLP)
# =========================
class DLOFeatureExtractor(BaseFeaturesExtractor):
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
        x = observations.view(observations.shape[0], -1).float()
        return self.net(x)

# =========================
# Builders
# =========================
def make_env(render: bool = False, seed: int = 42):
    # Gymnasium v0.29+: render_mode must be set at make()
    base = gym.make(
        "MountainCarContinuous-v0",
        render_mode=("rgb_array" if render else None),
    )
    base.reset(seed=seed)
    env = DLOWrapper(base)
    return env

# =========================
# Train / Eval
# =========================
def train(total_timesteps: int = 100_000, save_path: str = "ppo_hill_dlo.zip", seed: int = 42):
    print("Using cpu device")
    set_random_seed(seed)

    # Vectorized for PPO
    env = DummyVecEnv([lambda: make_env(render=False, seed=seed)])

    policy_kwargs = dict(
        features_extractor_class=DLOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,   # stable
        n_steps=4096,           # larger rollout
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        device="cpu",
    )

    print("🚀 Training PPO on MountainCarContinuous (D-LO)…")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

def evaluate(model_path: str = "ppo_hill_dlo.zip", episodes: int = 5, seed: int = 123, speed: str = "slow"):
    # Single env for clean render
    env = make_env(render=True, seed=seed)
    model = PPO.load(model_path)

    # Playback speed
    # slow: ~30 FPS, medium: ~45 FPS, fast: ~60 FPS
    delay = {"slow": 33, "medium": 22, "fast": 16}.get(speed.lower(), 33)

    print("🎮 AI Agent playing…  (Press 'q' to stop)\n")

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_return += float(reward)

                frame = env.render()
                if frame is not None:
                    cv2.imshow("D-LO Hill Climb", frame)
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopped by user.\n")
                        raise KeyboardInterrupt

            print(f"🏁 Episode {ep}: Return = {ep_return:.2f}")
            time.sleep(0.6)

    except KeyboardInterrupt:
        print("🛑 Evaluation stopped manually.")

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
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--model", type=str, default="ppo_hill_dlo.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=str, default="slow", choices=["slow", "medium", "fast"])
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, save_path=args.model, seed=args.seed)
    else:
        evaluate(model_path=args.model, seed=args.seed, speed=args.speed)
