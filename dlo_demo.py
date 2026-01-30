# dlo_demo.py
# Minimal end-to-end D-LO prototype:
# - Simple grid game (tile-based)
# - Env wrapper producing GO + three LOs (5x5,7x7,9x9) encoded as one-hot tile maps
# - SB3 PPO training with a custom PyTorch feature extractor that fuses LOs via attention
#
# Usage:
#   python dlo_demo.py train   # trains a small agent
#   python dlo_demo.py eval    # runs deterministic episodes and prints score

import argparse, math, random, os
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import warnings
warnings.filterwarnings("ignore")


# ---------------------
# D-LO wrapper: produce GO + LO_small/med/large tile-vector one-hot maps
# ---------------------
import cv2
import numpy as np
import time

class DLOWrapper(gym.Wrapper):
    def __init__(self, env, small=5, med=7, large=9):
        super().__init__(env)
        self.env = env
        self.small = small
        self.med = med
        self.large = large

        # Properly expanded observation space
        low = np.repeat(env.observation_space.low.flatten(), 3)
        high = np.repeat(env.observation_space.high.flatten(), 3)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.zoom_state = "global"
        self.alert_text = ""
        self.alert_color = (0, 255, 0)
        self.last_alert_time = 0

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32).flatten()

        # Multi-scale fusion
        small_view = obs * 0.5
        medium_view = obs
        large_view = obs * 2.0

        # Zoom logic (based on distance to pipe)
        distance_to_pipe = obs[0] if len(obs) > 0 else 0
        current_time = time.time()
        if distance_to_pipe < 0.05 and self.zoom_state != "local":
            self.zoom_state = "local"
            self.alert_text = "🚨 Obstacle Ahead — Zooming In"
            self.alert_color = (0, 0, 255)
            self.last_alert_time = current_time
        elif distance_to_pipe > 0.25 and self.zoom_state != "global":
            self.zoom_state = "global"
            self.alert_text = "✅ Obstacle Passed — Zooming Out"
            self.alert_color = (0, 255, 0)
            self.last_alert_time = current_time

        # Fuse features (dual observation)
        fused = np.concatenate([small_view, medium_view, large_view])
        return fused

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.zoom_state = "global"
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 🧠 Reward shaping — makes learning much faster
        reward += 0.1  # survival reward
        if terminated:
            reward -= 5.0  # big penalty for dying

        return self.observation(obs), reward, terminated, truncated, info

    def render(self, mode="human"):
        frame = self.env.render()
        if frame is not None:
            h, w, _ = frame.shape
            if self.zoom_state == "local":
                zoom_factor = 1.3
                cx, cy = w // 2, h // 2
                nw, nh = int(w / zoom_factor), int(h / zoom_factor)
                x1, y1 = cx - nw // 2, cy - nh // 2
                x2, y2 = cx + nw // 2, cy + nh // 2
                cropped = frame[y1:y2, x1:x2]
                frame = cv2.resize(cropped, (w, h))
            else:
                # Slight zoom-out
                zoom_factor = 0.95
                resized = cv2.resize(frame, (int(w * zoom_factor), int(h * zoom_factor)))
                frame = np.zeros_like(frame)
                y_off = (h - resized.shape[0]) // 2
                x_off = (w - resized.shape[1]) // 2
                frame[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

            # HUD overlay
            cv2.putText(frame, self.alert_text, (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.alert_color, 3)
            cv2.imshow("Flappy Bird - DLO Agent", frame)
            cv2.waitKey(1)
        return frame

# ---------------------
# PyTorch feature extractor with attention fusion
# ---------------------
class DLOFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        obs = observations.float().view(observations.shape[0], -1)
        return self.net(obs)

# ---------------------
# Training and Evaluation Functions
# ---------------------
def make_env(render=False):
    render_mode = "human" if render else None
    env = gym.make("FlappyBird-v0", render_mode=render_mode)
    env = DLOWrapper(env, small=5, med=7, large=9)
    return env

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def train(total_timesteps=1000000, save_path="ppo_flappy_dlo.zip", seed=42):
    env = DummyVecEnv([lambda: make_env(render=False)])
    env = VecFrameStack(env, n_stack=4)

    set_random_seed(seed)

    policy_kwargs = dict(
        features_extractor_class=DLOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        device="cpu"
    )

    print("🚀 Starting PPO training with D-LO fusion...")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

# ---------------------
# Evaluation (no average score)
# ---------------------
import numpy as np
import cv2
import time

def evaluate(model_path="ppo_flappy_dlo.zip", episodes=5):
    env = DummyVecEnv([lambda: make_env(render=True)])
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load(model_path)

    print("🎮 AI Agent playing... (Press Q in the window or Ctrl+C in terminal to stop)\n")

    try:
        for episode_count in range(1, episodes + 1):
            obs = env.reset()
            done = False
            total_r = 0
            zoomed_in = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step([int(action[0])])
                total_r += float(reward)

                frame = env.render(mode="rgb_array")
                bird_y = obs[0][1]
                pipe_y = obs[0][3] if len(obs[0]) > 3 else bird_y
                distance = abs(bird_y - pipe_y)

                if distance < 0.1 and not zoomed_in:
                    print("⚠️ Obstacle ahead — zooming in!")
                    zoomed_in = True
                elif distance >= 0.1 and zoomed_in:
                    print("✅ Obstacle cleared — zooming out!")
                    zoomed_in = False

                if frame is not None:
                    frame = np.array(frame)
                    cv2.putText(frame, f"Episode: {episode_count} | Reward: {total_r:.1f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if zoomed_in:
                        cv2.putText(frame, "⚠️ ZOOMING IN!", (10, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "✅ NORMAL VIEW", (10, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("D-LO Flappy Bird", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopping game (Q pressed).")
                        raise KeyboardInterrupt

            print(f"🏁 Episode {episode_count}: Total Reward = {total_r:.2f}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Game stopped manually by user.")

    finally:
        env.close()
        cv2.destroyAllWindows()
        time.sleep(0.5)

# ---------------------
# Command-line interface
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--model", type=str, default="ppo_flappy_dlo.zip")
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, save_path=args.model)
    elif args.mode == "eval":
        evaluate(model_path=args.model)
