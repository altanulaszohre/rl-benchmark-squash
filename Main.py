"""
Title: Off-Policy and On-Policy Deep Reinforcement Learning in a Pygame-Based Squash Environment
This project was developed by Altan Ulaş Zöhre and Deniz Sakaroğlu.
License: MIT License
"""

import gymnasium as gym
import numpy as np
import pygame
import random
import torch as th
import pandas as pd
from gymnasium import spaces

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# =========================
# Global experiment knobs
# =========================
BUDGET = 50_000          # all models train with same env steps
EVAL_EPISODES = 50      # fixed evaluation episodes per model
GLOBAL_SEED = 42         # base seed


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def render_fit(font, text, color, max_w):
    """Render text but clip with ellipsis if it exceeds max_w."""
    if font.size(text)[0] <= max_w:
        return font.render(text, True, color)

    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if font.size(cand)[0] <= max_w:
            lo = mid + 1
        else:
            hi = mid
    clipped = text[:max(0, lo - 1)] + ell
    return font.render(clipped, True, color)


class SquashEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SquashEnv, self).__init__()
        self.width = 280
        self.height = 380
        self.paddle_width = 60
        self.paddle_height = 12
        self.ball_radius = 7
        self.render_mode = render_mode

        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PADDLE = (0, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BORDER = (80, 80, 100)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.score = 0

        self.ball_dx = random.choice([-3, -2, 2, 3])
        self.ball_dy = -3
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.paddle_x / self.width,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / 10.0,
            self.ball_dy / 10.0
        ], dtype=np.float32)

    def step(self, action):
        prev_dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)

        speed = 8
        if action == 1:
            self.paddle_x -= speed
        elif action == 2:
            self.paddle_x += speed
        self.paddle_x = np.clip(self.paddle_x, 0, self.width - self.paddle_width)

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        reward = 0.0
        terminated = False

        # dense shaping: move paddle closer to ball-x
        current_dist = abs((self.paddle_x + self.paddle_width / 2) - self.ball_x)
        reward += 0.5 if current_dist < prev_dist else -0.5
        reward += 0.05  # living reward

        # wall bounces
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_dy *= -1

        # paddle collision near bottom band
        if (self.ball_y >= self.height - 20 and self.ball_y <= self.height - 5 and self.ball_dy > 0):
            if self.paddle_x - 10 < self.ball_x < self.paddle_x + self.paddle_width + 10:
                self.ball_dy *= -1
                self.score += 1
                reward += 15.0
                if abs(self.ball_dx) < 9:
                    self.ball_dx *= 1.05
                    self.ball_dy *= 1.05

        # miss -> terminate
        if self.ball_y > self.height:
            terminated = True
            reward -= 15.0

        return self._get_obs(), reward, terminated, False, {"score": self.score}

    def render(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.COLOR_BG)
        pygame.draw.rect(canvas, self.COLOR_BORDER, (0, 0, self.width, self.height), 2)
        pygame.draw.rect(
            canvas, self.COLOR_PADDLE,
            (self.paddle_x, self.height - 20, self.paddle_width, self.paddle_height),
            border_radius=4
        )
        pygame.draw.circle(canvas, self.COLOR_BALL, (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        return canvas


def _make_vec_env(seed: int = 0):
    def _thunk():
        env = SquashEnv()
        env = Monitor(env)          # adds info["episode"] = {"r":..., "l":..., "t":...}
        env.reset(seed=seed)
        return env
    return DummyVecEnv([_thunk])


# =========================
# Callback: log episode returns during training
# =========================
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, model_name: str, seed: int):
        super().__init__(verbose=0)
        self.model_name = model_name
        self.seed = seed
        self.rows = []

    def _on_step(self) -> bool:
        # VecEnv: infos is list of dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep = info["episode"]  # {"r": ep_return, "l": ep_length, "t": elapsed_time}
                self.rows.append({
                    "model": self.model_name,
                    "seed": self.seed,
                    "timesteps": int(self.num_timesteps),
                    "episode_return": float(ep.get("r", np.nan)),
                    "episode_length": int(ep.get("l", 0))
                })
        return True


# =========================
# Fixed evaluation (paper-grade)
# =========================
def evaluate_model(model, model_name: str, seed: int, n_episodes: int = 100):
    env = SquashEnv()
    scores, returns, lengths = [], [], []

    # make eval deterministic & reproducible-ish
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        total_r = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += float(r)
            steps += 1
            done = bool(terminated or truncated)

        scores.append(int(info.get("score", 0)))
        returns.append(float(total_r))
        lengths.append(int(steps))

    df_ep = pd.DataFrame({
        "model": model_name,
        "seed": seed,
        "episode": np.arange(1, n_episodes + 1),
        "score": scores,
        "return": returns,
        "length": lengths
    })

    summary = {
        "model": model_name,
        "seed": seed,
        "n_eval_episodes": n_episodes,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "best_score": int(np.max(scores)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths))
    }
    return df_ep, summary


def train_models_and_log():
    set_global_seeds(GLOBAL_SEED)

    print("\n" + "=" * 70)
    print("TRAINING: DQN-S vs DQN-L (tuned) vs PPO-S vs PPO-L (same budget)")
    print("=" * 70)

    trained = []
    training_rows = []
    hyper_rows = []

    # -------------------------
    # 1) DQN-S
    # -------------------------
    name = "DQN-S (128x128)"
    seed = 0
    env = _make_vec_env(seed=seed)

    dqn_policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
        policy_kwargs=dqn_policy_kwargs,
    )
    cb = EpisodeLoggerCallback(name, seed)
    model.learn(total_timesteps=BUDGET, callback=cb)

    trained.append((name, seed, model))
    training_rows.extend(cb.rows)
    hyper_rows.append({"model": name, "seed": seed, "algo": "DQN", "net_arch": "128,128",
                       "learning_rate": 1e-3, "buffer_size": 50_000, "learning_starts": 2000,
                       "batch_size": 64, "gamma": 0.99, "target_update_interval": 1000,
                       "exploration_fraction": 0.15, "exploration_final_eps": 0.02,
                       "budget_timesteps": BUDGET})

    # -------------------------
    # 2) DQN-L (tuned)
    # -------------------------
    name = "DQN-L (256x256 tuned)"
    seed = 1
    env = _make_vec_env(seed=seed)

    dqn_l_policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.995,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.10,
        exploration_final_eps=0.01,
        policy_kwargs=dqn_l_policy_kwargs,
    )
    cb = EpisodeLoggerCallback(name, seed)
    model.learn(total_timesteps=BUDGET, callback=cb)

    trained.append((name, seed, model))
    training_rows.extend(cb.rows)
    hyper_rows.append({"model": name, "seed": seed, "algo": "DQN", "net_arch": "256,256",
                       "learning_rate": 3e-4, "buffer_size": 100_000, "learning_starts": 5000,
                       "batch_size": 128, "gamma": 0.995, "target_update_interval": 500,
                       "exploration_fraction": 0.10, "exploration_final_eps": 0.01,
                       "budget_timesteps": BUDGET})

    # -------------------------
    # 3) PPO-S
    # -------------------------
    name = "PPO-S (128x128)"
    seed = 2
    env = _make_vec_env(seed=seed)

    ppo_policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=ppo_policy_kwargs,
    )
    cb = EpisodeLoggerCallback(name, seed)
    model.learn(total_timesteps=BUDGET, callback=cb)

    trained.append((name, seed, model))
    training_rows.extend(cb.rows)
    hyper_rows.append({"model": name, "seed": seed, "algo": "PPO", "net_arch": "128,128",
                       "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 256, "n_epochs": 10,
                       "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                       "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
                       "budget_timesteps": BUDGET})

    # -------------------------
    # 4) PPO-L (NEW)
    # -------------------------
    name = "PPO-L (256x256)"
    seed = 3
    env = _make_vec_env(seed=seed)

    ppo_l_policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=ppo_l_policy_kwargs,
    )
    cb = EpisodeLoggerCallback(name, seed)
    model.learn(total_timesteps=BUDGET, callback=cb)

    trained.append((name, seed, model))
    training_rows.extend(cb.rows)
    hyper_rows.append({"model": name, "seed": seed, "algo": "PPO", "net_arch": "256,256",
                       "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 256, "n_epochs": 10,
                       "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                       "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
                       "budget_timesteps": BUDGET})

    # =========================
    # Save training logs
    # =========================
    df_train = pd.DataFrame(training_rows)
    df_train.to_csv("training_episodes.csv", index=False)
    print("Saved: training_episodes.csv")

    df_hyp = pd.DataFrame(hyper_rows)
    df_hyp.to_csv("hyperparams.csv", index=False)
    print("Saved: hyperparams.csv")

    # =========================
    # Fixed evaluation + save
    # =========================
    eval_ep_frames = []
    eval_summaries = []
    for (mname, mseed, m) in trained:
        df_ep, summ = evaluate_model(m, mname, mseed, n_episodes=EVAL_EPISODES)
        eval_ep_frames.append(df_ep)
        eval_summaries.append(summ)

    pd.concat(eval_ep_frames, ignore_index=True).to_csv("eval_episodes.csv", index=False)
    pd.DataFrame(eval_summaries).to_csv("eval_summary.csv", index=False)
    print("Saved: eval_episodes.csv")
    print("Saved: eval_summary.csv")

    return trained


def run_dashboard(trained_models):
    models = [m for (_, _, m) in trained_models]
    names = [n for (n, _, _) in trained_models]

    envs = [SquashEnv() for _ in range(len(models))]
    obs_list = [env.reset()[0] for env in envs]
    scores_history = {name: [] for name in names}

    pygame.init()

    GAME_W, GAME_H = 280, 380
    PAD = 20
    HEADER = 90
    N = len(models)

    TOTAL_W = (GAME_W * N) + (PAD * (N + 1))
    TOTAL_H = GAME_H + HEADER + PAD
    screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
    pygame.display.set_caption("RL Benchmark — DQN vs PPO (Same Budget)")
    clock = pygame.time.Clock()

    font_header = pygame.font.SysFont("Verdana", 22, bold=True)
    font_sub = pygame.font.SysFont("Consolas", 15, bold=True)

    title_colors = [(235, 80, 80), (240, 200, 60), (80, 230, 100), (120, 140, 255)]
    panel_w = GAME_W
    max_text_w = panel_w

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((15, 15, 20))

        for i in range(N):
            action, _ = models[i].predict(obs_list[i], deterministic=True)
            obs, reward, terminated, _, info = envs[i].step(int(action))
            obs_list[i] = obs

            if terminated:
                scores_history[names[i]].append(info["score"])
                obs_list[i], _ = envs[i].reset()

            x_pos = PAD + (i * (GAME_W + PAD))
            y_pos = HEADER

            history = scores_history[names[i]]
            avg = float(np.mean(history)) if history else 0.0
            best = int(np.max(history)) if history else 0
            live = envs[i].score
            eps = len(history)

            title_surf = render_fit(font_header, names[i], title_colors[i % len(title_colors)], max_text_w)
            screen.blit(title_surf, (x_pos, 12))

            line1 = f"Live: {live} | Best: {best}"
            line2 = f"Avg: {avg:.1f} | Episodes: {eps}"

            s1 = render_fit(font_sub, line1, (200, 200, 200), max_text_w)
            s2 = render_fit(font_sub, line2, (200, 200, 200), max_text_w)
            screen.blit(s1, (x_pos, 45))
            screen.blit(s2, (x_pos, 62))

            game_surface = envs[i].render()
            screen.blit(game_surface, (x_pos, y_pos))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # Save dashboard-run episode scores (optional extra)
    print("Dashboard closed. Saving dashboard scores...")
    max_len = max(len(v) for v in scores_history.values()) if scores_history else 0
    df_dict = {}
    for name, scores in scores_history.items():
        df_dict[name] = scores + [np.nan] * (max_len - len(scores))
    pd.DataFrame(df_dict).to_csv("dashboard_scores.csv", index=False)
    print("Saved: dashboard_scores.csv")


if __name__ == "__main__":
    trained = train_models_and_log()
    run_dashboard(trained)
