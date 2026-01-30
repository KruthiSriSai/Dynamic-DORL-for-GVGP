# subway_dlo.py
# Pygame runner with Dual (Global+Local) Observation (D-LO) + PPO (SB3)
# - 3 lanes, oncoming obstacles & coins, speed ramps up
# - Observation: low-dim state -> D-LO wrapper expands to 3x scales (0.5x,1x,2x)
# - Pygame window during eval (press 'Q' to quit)
#
# Train:
#   python subway_dlo.py train --timesteps 120000 --model ppo_subway_dlo.zip
# Eval:
#   python subway_dlo.py eval   --model ppo_subway_dlo.zip

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pygame

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# -----------------------
# Core game datatypes
# -----------------------
LANES = 3
LANE_IDX = [0, 1, 2]

@dataclass
class Obstacle:
    lane: int
    y: float   # distance ahead (down-screen)
    kind: int  # 0: low block (jump), 1: tall block (lane change)
    speed: float

@dataclass
class Coin:
    lane: int
    y: float
    speed: float

# -----------------------
# Subway Runner Environment (Gymnasium)
# -----------------------
class SubwayRunnerEnv(gym.Env):
    """
    Action space: Discrete(3)
      0 = stay, 1 = move left, 2 = move right
    (Simple: jump is automatic if needed; the policy learns lane switching.)

    Base observation (vector, before D-LO):
      [ player_lane_onehot(3),
        v_norm,
        d_obs1, lane_obs1_onehot(3), kind_obs1_onehot(2),
        d_obs2, lane_obs2_onehot(3), kind_obs2_onehot(2),
        d_coin1, lane_coin1_onehot(3),
        d_coin2, lane_coin2_onehot(3) ]

    Rewards:
      +0.05 per step survived
      +1.0 per coin
      +0.5 per obstacle passed
      -5.0 on collision (episode ends)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = 0, render: bool = False):
        super().__init__()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # positions
        self.player_lane = 1  # middle
        self.player_y = 0.0   # not used for jumping arc; visual only

        # speeds & spawn
        self.speed = 7.0
        self.t = 0
        self.max_length = 4000  # step cap

        # gameplay lists
        self.obstacles: List[Obstacle] = []
        self.coins: List[Coin] = []

        # spawn pacing
        self.spawn_obs_timer = 0.0
        self.spawn_coin_timer = 0.0

        # observation shape (see docstring): compute dimensions
        self.obs_dim = 3 + 1 + (1 + 3 + 2) * 2 + (1 + 3) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # rendering
        self.render_enabled = render
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.quit_requested = False

        # scoring
        self.score = 0
        self.passed = 0

    def seed(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

    # ---------- helpers ----------
    def _reset_level(self):
        self.player_lane = 1
        self.player_y = 0.0
        self.speed = 7.0
        self.t = 0
        self.obstacles.clear()
        self.coins.clear()
        self.spawn_obs_timer = 0.0
        self.spawn_coin_timer = 0.0
        self.score = 0
        self.passed = 0

    def _spawn_obstacle(self):
        lane = self.rng.choice(LANE_IDX)
        kind = 0 if self.rng.random() < 0.6 else 1  # more low blocks than tall
        start_y = 40.0
        self.obstacles.append(Obstacle(lane=lane, y=start_y, kind=kind, speed=self.speed))

    def _spawn_coin(self):
        lane = self.rng.choice(LANE_IDX)
        start_y = 35.0
        self.coins.append(Coin(lane=lane, y=start_y, speed=self.speed))

    def _lane_onehot(self, lane: int) -> List[float]:
        return [1.0 if i == lane else 0.0 for i in LANE_IDX]

    def _kind_onehot(self, kind: int) -> List[float]:
        return [1.0 if kind == 0 else 0.0, 1.0 if kind == 1 else 0.0]

    def _nearest_two(self, objs, attr="y"):
        arr = sorted(objs, key=lambda o: getattr(o, attr))
        return (arr[0] if len(arr) > 0 else None), (arr[1] if len(arr) > 1 else None)

    def _build_obs(self) -> np.ndarray:
        # nearest 2 obstacles
        o1, o2 = self._nearest_two([o for o in self.obstacles if o.y >= 0.0])
        # nearest 2 coins
        c1, c2 = self._nearest_two([c for c in self.coins if c.y >= 0.0])

        def encode_obs(o: Optional[Obstacle]):
            if o is None:
                return [40.0] + [0.0, 0.0, 0.0] + [0.0, 0.0]
            return [float(o.y)] + self._lane_onehot(o.lane) + self._kind_onehot(o.kind)

        def encode_coin(c: Optional[Coin]):
            if c is None:
                return [40.0] + [0.0, 0.0, 0.0]
            return [float(c.y)] + self._lane_onehot(c.lane)

        v_norm = self.speed / 20.0
        vec = (
            self._lane_onehot(self.player_lane) +
            [v_norm] +
            encode_obs(o1) + encode_obs(o2) +
            encode_coin(c1) + encode_coin(c2)
        )
        return np.array(vec, dtype=np.float32)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        self._reset_level()
        obs = self._build_obs()
        info = {"score": self.score, "passed": self.passed}
        return obs, info

    def step(self, action: int):
        # apply action: lane change
        if action == 1 and self.player_lane > 0:
            self.player_lane -= 1
        elif action == 2 and self.player_lane < LANES - 1:
            self.player_lane += 1

        # move world
        dt = 1.0 / 30.0
        self.t += 1
        self.speed = min(20.0, self.speed + 0.002)  # gentle ramp-up

        # advance obstacles/coins
        for o in self.obstacles:
            o.y -= o.speed * dt
        for c in self.coins:
            c.y -= c.speed * dt

        # cleanup behind player
        self.obstacles = [o for o in self.obstacles if o.y > -2.0]
        self.coins     = [c for c in self.coins if c.y > -2.0]

        # spawn pacing
        self.spawn_obs_timer += dt
        self.spawn_coin_timer += dt
        if self.spawn_obs_timer > max(0.7, 1.6 - self.speed * 0.05):
            self.spawn_obs_timer = 0.0
            self._spawn_obstacle()
        if self.spawn_coin_timer > 0.9:
            self.spawn_coin_timer = 0.0
            if self.rng.random() < 0.7:
                self._spawn_coin()

        # collisions and passes
        reward = 0.05  # survival
        terminated = False
        truncated = False

        # coins
        for c in list(self.coins):
            if 0.0 < c.y < 1.0 and c.lane == self.player_lane:
                self.coins.remove(c)
                self.score += 1
                reward += 1.0

        # obstacles: pass or collide
        for o in list(self.obstacles):
            if 0.0 < o.y < 0.8:
                # overlapping region near player
                if o.lane == self.player_lane:
                    # tall kind (1) must change lane; low kind (0) auto-jump if "close enough"
                    if o.kind == 1:
                        reward -= 5.0
                        terminated = True
                        break
                    else:
                        # emulate a tiny auto-jump probability if exactly aligned: fail sometimes
                        if self.rng.random() < 0.08:
                            reward -= 5.0
                            terminated = True
                            break
                else:
                    # passed safely if just crossing under player
                    pass

            if o.y < 0.0 and not terminated:
                # counted as passed when it goes behind player
                self.obstacles.remove(o)
                self.passed += 1
                reward += 0.5

        if self.t >= self.max_length:
            truncated = True

        obs = self._build_obs()
        info = {"score": self.score, "passed": self.passed}
        return obs, reward, terminated, truncated, info

    # ---------- Pygame rendering ----------
    def _ensure_pygame(self):
        if self.screen is not None:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((520, 720))
        pygame.display.set_caption("Subway Runner — D-LO")
        self.clock = pygame.time.Clock()

    def render(self):
        self._ensure_pygame()
        assert self.screen is not None and self.clock is not None

        # event handling (quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.quit_requested = True

        # draw
        W, H = self.screen.get_size()
        self.screen.fill((18, 18, 24))

        # lanes
        lane_x = [int(W * 0.2), int(W * 0.5), int(W * 0.8)]
        for i, x in enumerate(lane_x):
            pygame.draw.line(self.screen, (60, 60, 80), (x, 0), (x, H), 2)
            if i < 2:
                pygame.draw.line(self.screen, (35, 35, 55),
                                 ((lane_x[i] + lane_x[i+1]) // 2, 0),
                                 ((lane_x[i] + lane_x[i+1]) // 2, H), 1)

        # player
        px = lane_x[self.player_lane]
        py = int(H * 0.85)
        pygame.draw.circle(self.screen, (80, 200, 255), (px, py), 18)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 18, 2)

        # world -> screen
        def world_to_screen(y):
            # y in [0..40] maps to screen along vertical
            return int(H * 0.85 - (y * (H * 0.02)))

        # obstacles
        for o in self.obstacles:
            x = lane_x[o.lane]
            y = world_to_screen(o.y)
            color = (255, 110, 90) if o.kind == 1 else (255, 180, 90)
            w = 44
            h = 28 if o.kind == 0 else 60
            pygame.draw.rect(self.screen, color, pygame.Rect(x - w//2, y - h, w, h), border_radius=6)

        # coins
        for c in self.coins:
            x = lane_x[c.lane]
            y = world_to_screen(c.y)
            pygame.draw.circle(self.screen, (250, 225, 40), (x, y), 10)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 10, 2)

        # D-LO “zoom” cue: highlight border when nearest obstacle within 6m
        near_dist = min([o.y for o in self.obstacles], default=99.0)
        if near_dist < 6.0:
            col = (255, 70, 70)
            thick = 6
        else:
            col = (80, 180, 100)
            thick = 3
        pygame.draw.rect(self.screen, col, pygame.Rect(8, 8, W-16, H-16), thick, border_radius=12)

        # HUD
        font = pygame.font.SysFont("Arial", 18)
        hud = font.render(f"Score: {self.score}   Passed: {self.passed}   Speed: {self.speed:.1f}", True, (230, 230, 240))
        self.screen.blit(hud, (12, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def get_quit(self) -> bool:
        return self.quit_requested

    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.screen = None
        self.clock = None
        self.quit_requested = False

# -----------------------
# D-LO Wrapper (3x scaled features)
# -----------------------
class DLOWrap(BaseFeaturesExtractor):
    """
    We’ll implement D-LO as the FeaturesExtractor for simplicity:
    input: base obs (Box, dim N)
    output: fused 3N = [0.5x, 1x, 2x] scales
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        fused_dim = int(np.prod(observation_space.shape)) * 3
        # init BaseFeaturesExtractor with final features_dim (we’ll produce fused then MLP)
        super().__init__(spaces.Box(low=-np.inf, high=np.inf, shape=(fused_dim,), dtype=np.float32),
                         features_dim)
        self.input_dim = fused_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Linear(256, features_dim),   nn.ReLU(),
        )

    def _fuse(self, obs: th.Tensor) -> th.Tensor:
        # obs shape: [B, N]
        s = obs * 0.5
        m = obs
        l = obs * 2.0
        return th.cat([s, m, l], dim=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations is actually the *base* obs; fuse here
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        fused = self._fuse(observations.float())
        return self.mlp(fused)

# -----------------------
# Helper: build envs
# -----------------------
def make_env(seed=0, render=False):
    env = SubwayRunnerEnv(seed=seed, render=render)
    return env

# -----------------------
# Train
# -----------------------
def train(total_timesteps=120000, save_path="ppo_subway_dlo.zip", seed=0):
    set_random_seed(seed)

    # Wrap base Box(N) with our D-LO features via policy_kwargs
    env = DummyVecEnv([lambda: make_env(seed=seed, render=False)])

    policy_kwargs = dict(
        features_extractor_class=DLOWrap,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        device="cpu",
    )

    print("🚀 Training PPO on Subway Runner (D-LO)…")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

# -----------------------
# Eval (real pygame window; press Q to quit)
# -----------------------
def evaluate(model_path="ppo_subway_dlo.zip", episodes=5, seed=0):
    # Use a single *non-vectorized* env for smooth real-time rendering
    env = make_env(seed=seed, render=True)
    model = PPO.load(model_path)

    print("🎮 AI Agent playing…  (Press 'q' to stop)\n")

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                # SB3 can handle single obs of shape (N,) for predict
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                ep_return += float(reward)

                env.render()
                if env.get_quit():
                    print("\n🛑 Stopped by user.")
                    return

            print(f"🏁 Episode {ep}: Return = {ep_return:.2f}, Score = {info.get('score', 0)}, Passed = {info.get('passed', 0)}")
            time.sleep(0.6)

    finally:
        env.close()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--timesteps", type=int, default=120000)
    parser.add_argument("--model", type=str, default="ppo_subway_dlo.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, save_path=args.model, seed=args.seed)
    else:
        evaluate(model_path=args.model, seed=args.seed)
