# jetpack_dlo.py
# Jetpack Joyride–style side scroller with Dual Local/Global Observations (D-LO)
# - Clean visuals via OpenCV (no text spam, MINIMAL HUD: green border = global, red = local)
# - Stable reward shaping for fast learning
# - Works like your Flappy script: train/eval + 'q' to quit eval

import argparse, math, random, time
from typing import List, Tuple, Deque, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import torch as th
import torch.nn as nn
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# ------------------------------
# Jetpack Env (custom, lightweight)
# ------------------------------
class JetpackEnv(gym.Env):
    """
    Simple Jetpack-style env:
      - State (for policy): [y, vy, next_gap_dx, next_gap_y, next_gap_h, laser_timer]
      - Action: 0 = no thrust, 1 = thrust up
      - Reward: +1 per step survived, +5 for passing a gate, -10 crash
      - Episode ends on crash or time limit
      - Rendering: smooth 800x480 window with moving obstacles
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, render_scale: Tuple[int, int] = (800, 480),
                 gravity: float = 0.18, thrust: float = 0.32,
                 gate_interval: int = 65, max_steps: int = 3500):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.W, self.H = render_scale
        self.gravity = gravity
        self.thrust = thrust
        self.gate_interval = gate_interval
        self.max_steps = max_steps

        # Player
        self.player_x = int(self.W * 0.18)
        self.player_y = self.H // 2
        self.player_r = 12

        # Obstacles: vertical gaps (“gates”) + occasional lasers
        self.gates: Deque[Dict] = deque()
        self.lasers: Deque[Dict] = deque()
        self.scroll_speed = 5

        # Action / Observation
        self.action_space = spaces.Discrete(2)  # 0 no-thrust, 1 thrust
        # low/high are loose, policy learns dynamics
        # [y, vy, next_dx, next_gap_y, next_gap_h, laser_timer]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, 0.0, 0.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([self.H, 5.0, self.W, self.H, 200.0, 200.0], dtype=np.float32),
            dtype=np.float32
        )

        self.reset(seed=seed)

    def _spawn_gate(self):
        gap_h = self.rng.randint(110, 160)       # gap height
        gap_y = self.rng.randint(80, self.H-80)  # gap center
        x = self.W + 40
        gate = dict(x=x, gap_y=gap_y, gap_h=gap_h, w=40, passed=False)
        self.gates.append(gate)

    def _maybe_spawn_laser(self):
        # random thin laser that forces minor vertical correction
        if self.rng.rand() < 0.18:
            y = self.rng.randint(60, self.H-60)
            # short burst laser
            self.lasers.append(dict(x0=self.W+20, x1=self.W+120, y=y, on=True, ttl=60))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.player_y = self.H // 2
        self.player_vy = 0.0
        self.steps = 0
        self.score = 0
        self.gates.clear()
        self.lasers.clear()
        self._spawn_gate()
        self.gate_cd = self.gate_interval
        obs = self._get_obs()
        info = {"score": self.score, "passed": 0, "coins": 0}
        return obs, info

    def _get_obs(self):
        # Pick the next gate ahead of player
        next_dx = float(self.W)
        ngy = float(self.H // 2)
        ngh = 140.0
        for g in self.gates:
            if g["x"] + g["w"] >= self.player_x:
                next_dx = float(g["x"] + g["w"] - self.player_x)
                ngy = float(g["gap_y"])
                ngh = float(g["gap_h"])
                break

        # Laser timer (time to any active laser crossing vertical)
        laser_timer = 0.0
        if len(self.lasers) > 0:
            laser_timer = float(self.lasers[0]["ttl"])

        return np.array(
            [self.player_y, self.player_vy, next_dx, ngy, ngh, laser_timer],
            dtype=np.float32
        )

    def step(self, action: int):
        # physics
        if action == 1:
            self.player_vy -= self.thrust
        self.player_vy += self.gravity
        self.player_vy = np.clip(self.player_vy, -4.0, 4.0)
        self.player_y += int(self.player_vy)

        # scroll world
        for g in list(self.gates):
            g["x"] -= self.scroll_speed
        for lz in list(self.lasers):
            lz["x0"] -= self.scroll_speed
            lz["x1"] -= self.scroll_speed
            lz["ttl"] -= 1
            if lz["ttl"] <= 0:
                self.lasers.popleft()

        # spawn
        self.gate_cd -= 1
        if self.gate_cd <= 0:
            self._spawn_gate()
            self._maybe_spawn_laser()
            self.gate_cd = self.gate_interval

        # gate passing & cleanup
        passed_now = 0
        for g in list(self.gates):
            if g["x"] + g["w"] < self.player_x and not g["passed"]:
                g["passed"] = True
                passed_now += 1
            if g["x"] + g["w"] < -10:
                self.gates.popleft()

        # collisions
        crash = False
        # bounds
        if self.player_y - self.player_r < 0 or self.player_y + self.player_r >= self.H:
            crash = True
        # gate collision
        for g in self.gates:
            if self.player_x in range(g["x"], g["x"] + g["w"]):
                gap_top = g["gap_y"] - g["gap_h"] // 2
                gap_bot = g["gap_y"] + g["gap_h"] // 2
                if not (gap_top <= self.player_y <= gap_bot):
                    crash = True
                    break
        # laser collision
        for lz in self.lasers:
            if lz["on"]:
                if lz["x0"] - 2 <= self.player_x <= lz["x1"] + 2 and abs(self.player_y - lz["y"]) <= 8:
                    crash = True
                    break

        # reward shaping
        reward = 1.0  # survival reward
        if passed_now > 0:
            reward += 5.0 * passed_now
            self.score += passed_now
        if crash:
            reward -= 10.0

        self.steps += 1
        terminated = bool(crash)
        truncated = bool(self.steps >= self.max_steps)
        obs = self._get_obs()
        info = {"score": self.score, "passed": passed_now, "coins": 0}
        return obs, reward, terminated, truncated, info

    def render(self):
        # draw world (MINIMAL HUD)
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # background
        frame[:] = (10, 12, 18)
        # subtle stars
        for _ in range(40):
            x = self.rng.randint(0, self.W)
            y = self.rng.randint(0, self.H)
            frame[y, x] = (22, 24, 36)

        # draw gates
        for g in self.gates:
            x = g["x"]
            w = g["w"]
            gap_y = g["gap_y"]
            gap_h = g["gap_h"]
            top = gap_y - gap_h // 2
            bot = gap_y + gap_h // 2
            # upper block
            cv2.rectangle(frame, (x, 0), (x + w, max(0, top)), (50, 170, 220), -1)
            # lower block
            cv2.rectangle(frame, (x, min(self.H, bot)), (x + w, self.H), (50, 170, 220), -1)

        # draw lasers
        for lz in self.lasers:
            if lz["on"]:
                cv2.line(frame, (lz["x0"], lz["y"]), (lz["x1"], lz["y"]), (40, 230, 40), 3)

        # draw player
        cv2.circle(frame, (self.player_x, self.player_y), self.player_r, (220, 210, 60), -1)

        return frame

# ------------------------------
# D-LO Wrapper (MINIMAL HUD)
# ------------------------------
class DLOWrapper(gym.Wrapper):
    """
    Dual Local/Global feature vector + minimal render zoom:
      - Observation (Box, 6): we create three scales small/med/large: concat -> 18 dim
      - Local condition: if next gate is near (dx < 110) or wall/laser proximity
      - Minimal HUD: red border (local) / green border (global)
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box) and env.observation_space.shape == (6,)
        base_low = env.observation_space.low.astype(np.float32)
        base_high = env.observation_space.high.astype(np.float32)
        low = np.concatenate([base_low * 0.5, base_low, base_low * 2.0]).astype(np.float32)
        high = np.concatenate([base_high * 0.5, base_high, base_high * 2.0]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.zoom_state = "global"

    def _fuse(self, obs: np.ndarray) -> np.ndarray:
        s = obs.astype(np.float32)
        return np.concatenate([s * 0.5, s, s * 2.0], axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        fused = self._fuse(obs)
        self.zoom_state = "global"
        return fused, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        fused = self._fuse(obs)
        # local trigger heuristic
        y, vy, dx, gy, gh, ltime = obs
        local = (dx < 110.0) or (y < 25.0) or (y > (self.env.H - 25.0)) or (ltime > 0.0 and dx < 160.0)
        self.zoom_state = "local" if local else "global"
        # Small survival shaping (keeps learning brisk)
        reward += 0.10
        if terminated:
            reward -= 3.0
        return fused, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        h, w, _ = frame.shape
        # zoom
        if self.zoom_state == "local":
            z = 1.25
            cx, cy = w // 2, h // 2
            nw, nh = int(w / z), int(h / z)
            x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
            x2, y2 = min(w, cx + nw // 2), min(h, cy + nh // 2)
            crop = frame[y1:y2, x1:x2]
            frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            z = 0.97
            resized = cv2.resize(frame, (int(w * z), int(h * z)))
            result = np.zeros_like(frame)
            y_off = (h - resized.shape[0]) // 2
            x_off = (w - resized.shape[1]) // 2
            result[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized
            frame = result
        # minimal border
        color = (40, 210, 60) if self.zoom_state == "global" else (50, 50, 230)
        cv2.rectangle(frame, (8, 8), (w-8, h-8), color, 3)
        return frame

# ------------------------------
# SB3 Feature Extractor (MLP)
# ------------------------------
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DLOFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))  # 18-dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU()
        )
    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.view(obs.shape[0], -1).float()
        return self.net(x)

# ------------------------------
# Helpers / Train / Eval
# ------------------------------
def make_env(seed: int = 0):
    base = JetpackEnv(seed=seed, render_scale=(800,480))
    base.render_mode = "rgb_array"     # 👈 IMPORTANT
    return DLOWrapper(base)


def train(total_timesteps: int = 180_000, save_path: str = "ppo_jetpack_dlo.zip", seed: int = 42):
    set_random_seed(seed)
    env = DummyVecEnv([lambda: make_env(seed)])
    policy_kwargs = dict(
        features_extractor_class=DLOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],
    )
    model = PPO(
        "MlpPolicy", env, verbose=1, seed=seed, policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4, n_steps=4096, batch_size=512, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001, device="cpu"
    )
    print("🚀 Training PPO on Jetpack (D-LO)…")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

def evaluate(model_path="ppo_jetpack_dlo.zip", episodes=5, seed: int = 42):
    env = DummyVecEnv([lambda: make_env(seed)])
    model = PPO.load(model_path)
    print("🎮 AI Agent playing…  (Press 'q' to stop)\n")
    try:
        for ep in range(1, episodes+1):
            obs = env.reset()
            done = False
            ep_return = 0.0
            score = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step([int(action)])  # vec env step
                ep_return += float(reward)

                if isinstance(info, list) and len(info) and "score" in info[0]:
                    score = info[0]["score"]

                frame = env.render(mode="rgb_array")

                if frame is not None:
                    cv2.imshow("Jetpack D-LO", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopped by user.")
                        raise KeyboardInterrupt
            print(f"🏁 Episode {ep}: Return = {ep_return:.2f}, Score = {score}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        env.close(); cv2.destroyAllWindows(); time.sleep(0.2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--timesteps", type=int, default=180000)
    p.add_argument("--model", type=str, default="ppo_jetpack_dlo.zip")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.mode == "train":
        train(total_timesteps=args.timesteps, save_path=args.model, seed=args.seed)
    else:
        evaluate(model_path=args.model, seed=args.seed)
