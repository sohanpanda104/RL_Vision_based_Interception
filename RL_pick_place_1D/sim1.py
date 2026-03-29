"""
Stage 1 - Baseline: Constant Velocity Interception
===================================================
A 2D hand must intercept and grasp a block sliding across a table
at a constant velocity. State-based observations (no camera yet).

Uses PPO from Stable-Baselines3 with GPU acceleration.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import os
import time


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
class ConveyorInterceptionEnv(gym.Env):
    """
    2D interception environment (Stage 1 - Constant Velocity Baseline).

    The hand starts on the left side. A block spawns on the right and
    slides leftward at constant speed. The agent must move the hand to
    intercept and close the gripper at the right moment.

    Observation (11-dim):
        hand_x, hand_y,          - hand position
        obj_x, obj_y,            - object position
        obj_vx, obj_vy,          - object velocity
        rel_x, rel_y,            - relative vector (obj - hand)
        distance,                - scalar distance
        future_rel_x, future_rel_y  - predicted intercept point offset

    Action (3-dim, continuous):
        dx, dy  in [-1, 1]       - hand movement direction
        grip    in [0, 1]        - gripper (>0.5 = close)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # --- Workspace ---
        self.x_min, self.x_max = -1.0, 1.0
        self.y_min, self.y_max = -1.0, 1.0

        # --- Hand dynamics ---
        # Hand is 2-5x faster than object so it can always catch up
        self.hand_speed = 0.14

        # --- Object dynamics ---
        self.obj_speed_min = 0.02
        self.obj_speed_max = 0.035

        # --- Success thresholds ---
        self.grasp_radius = 0.10     # gripper close reward zone
        self.success_radius = 0.08   # actual success check

        # --- Episode settings ---
        self.max_steps = 300

        # --- Tracking for reward shaping ---
        self.prev_dist = None
        self.prev_hand_pos = None

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low = np.array([
            self.x_min, self.y_min,     # hand
            self.x_min, self.y_min,     # object
            -1.0, -1.0,                 # velocity
            -3.0, -3.0,                 # relative
            0.0,                        # distance
            -3.0, -3.0                  # future intercept offset
        ], dtype=np.float32)

        obs_high = np.array([
            self.x_max, self.y_max,
            self.x_max, self.y_max,
            1.0, 1.0,
            3.0, 3.0,
            4.0,
            3.0, 3.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Runtime state
        self.hand_pos = None
        self.obj_pos = None
        self.obj_vel = None
        self.gripper_closed = False
        self.steps = 0

    # ----- helpers -----

    def _get_obs(self):
        rel = self.obj_pos - self.hand_pos
        dist = np.linalg.norm(rel)

        # Predict where the object will be in ~10 steps
        future_obj = self.obj_pos + self.obj_vel * 10
        future_rel = future_obj - self.hand_pos

        return np.array([
            self.hand_pos[0], self.hand_pos[1],
            self.obj_pos[0], self.obj_pos[1],
            self.obj_vel[0], self.obj_vel[1],
            rel[0], rel[1],
            dist,
            future_rel[0], future_rel[1],
        ], dtype=np.float32)

    def _clip(self, pos):
        pos[0] = np.clip(pos[0], self.x_min, self.x_max)
        pos[1] = np.clip(pos[1], self.y_min, self.y_max)
        return pos

    # ----- gym interface -----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.gripper_closed = False

        # Hand starts left-center area
        self.hand_pos = np.array([
            self.np_random.uniform(-0.4, 0.0),
            self.np_random.uniform(-0.3, 0.3),
        ], dtype=np.float32)

        # Object starts on the right edge
        self.obj_pos = np.array([
            self.np_random.uniform(0.65, 0.90),
            self.np_random.uniform(-0.3, 0.3),
        ], dtype=np.float32)

        # Constant leftward velocity (this is Stage 1)
        speed = self.np_random.uniform(self.obj_speed_min, self.obj_speed_max)
        vx = -speed
        vy = self.np_random.uniform(-0.003, 0.003)   # tiny drift
        self.obj_vel = np.array([vx, vy], dtype=np.float32)

        # Initialize tracking vars
        self.prev_dist = np.linalg.norm(self.obj_pos - self.hand_pos)
        self.prev_hand_pos = self.hand_pos.copy()

        return self._get_obs(), {}

    def step(self, action):
        dx = float(np.clip(action[0], -1.0, 1.0))
        dy = float(np.clip(action[1], -1.0, 1.0))
        grip = float(np.clip(action[2], 0.0, 1.0))

        self.steps += 1

        # --- Move hand ---
        move_vec = np.array([dx, dy], dtype=np.float32)
        move_norm = np.linalg.norm(move_vec)
        if move_norm > 1.0:
            move_vec = move_vec / move_norm  # normalize so diagonal isn't faster
        self.hand_pos = self._clip(self.hand_pos + move_vec * self.hand_speed)

        # --- Move object ---
        self.obj_pos = self._clip(self.obj_pos + self.obj_vel)

        # --- Compute distances ---
        rel = self.obj_pos - self.hand_pos
        dist = float(np.linalg.norm(rel))

        # ===================================================================
        #  REWARD FUNCTION
        # ===================================================================
        reward = 0.0

        # 1) Small time penalty — encourages faster interception
        reward -= 0.3

        # 2) Approach reward — reward for getting closer to the object
        #    This is the KEY shaping signal: positive when dist decreases
        dist_improvement = self.prev_dist - dist
        reward += dist_improvement * 50.0

        # 2b) Urgency bonus — stronger pull when object is near left edge
        #     Object x ranges from ~0.9 (start) to -1.0 (escape)
        #     When object is past center (x < 0), urgency kicks in
        if self.obj_pos[0] < 0.0 and dist > self.success_radius:
            urgency = max(0.0, 1.0 - (self.obj_pos[0] + 1.0))  # 0→1 as obj nears edge
            reward += dist_improvement * 30.0 * urgency  # extra approach bonus

        # 3) Proximity bonus — small bonus for being near the object
        if dist < 0.3:
            reward += 2.0 * (0.3 - dist)  # max +0.6 when on top

        # 4) Velocity matching bonus — when close, reward moving in same
        #    direction as object (so the hand doesn't knock it away)
        if dist < 0.2:
            hand_vel = self.hand_pos - self.prev_hand_pos
            hand_vel_norm = np.linalg.norm(hand_vel)
            obj_vel_norm = np.linalg.norm(self.obj_vel)
            if hand_vel_norm > 1e-6 and obj_vel_norm > 1e-6:
                cos_sim = np.dot(hand_vel, self.obj_vel) / (hand_vel_norm * obj_vel_norm)
                reward += 1.0 * max(0.0, cos_sim)  # 0 to +1

        # 5) Gripper rewards
        gripper_close = grip > 0.5

        if gripper_close and dist < self.grasp_radius:
            reward += 15.0   # good: closing when near
        elif gripper_close and dist > 0.25:
            reward -= 3.0    # bad: closing when far away

        # 6) SUCCESS — the big prize
        terminated = False
        truncated = False
        info = {"success": False}

        if gripper_close and dist < self.success_radius:
            reward += 500.0
            terminated = True
            info["success"] = True

        # 7) Object escapes left edge — moderate penalty
        if self.obj_pos[0] <= self.x_min + 0.01:
            reward -= 50.0
            terminated = True
            info["missed"] = True

        # 8) Max steps reached
        if self.steps >= self.max_steps:
            truncated = True

        # --- Update tracking ---
        self.prev_dist = dist
        self.prev_hand_pos = self.hand_pos.copy()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Step {self.steps:3d} | "
            f"hand=({self.hand_pos[0]:+.2f},{self.hand_pos[1]:+.2f}) | "
            f"obj=({self.obj_pos[0]:+.2f},{self.obj_pos[1]:+.2f}) | "
            f"dist={np.linalg.norm(self.obj_pos - self.hand_pos):.3f}"
        )


# ---------------------------------------------------------------------------
#  Utility: make env factory (needed for SubprocVecEnv)
# ---------------------------------------------------------------------------
def make_env(rank, seed=0):
    """Returns a function that creates and returns a monitored env."""
    def _init():
        env = ConveyorInterceptionEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ---------------------------------------------------------------------------
#  Detect GPU
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] Using: {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    else:
        print("[CPU] No CUDA GPU detected — training on CPU")
        return "cpu"


# ---------------------------------------------------------------------------
#  Train
# ---------------------------------------------------------------------------
def train(total_timesteps=3_000_000):
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    device = get_device()
    n_envs = 16   # parallel environments (CPU-side, doesn't use VRAM)

    print(f"[INFO] Training for {total_timesteps:,} timesteps across {n_envs} envs")
    print(f"[INFO] That's ~{total_timesteps // n_envs:,} steps per env")

    # --- Training env (parallel) ---
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # --- Eval env (single, for periodic testing) ---
    eval_env = DummyVecEnv([make_env(100)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # don't normalize eval rewards
        clip_obs=10.0,
    )
    eval_env.training = False

    # --- PPO Model ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        device=device,                 # GPU acceleration (RTX 4050)
        learning_rate=3e-4,
        n_steps=2048,                  # steps per env before update
        batch_size=512,                # larger batch = better GPU utilization
        n_epochs=10,
        gamma=0.995,                   # slightly higher for longer episodes
        gae_lambda=0.95,
        ent_coef=0.02,                 # more exploration
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model",
        log_path="models/logs",
        eval_freq=max(10_000 // n_envs, 1),   # eval every ~10K total steps
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),   # checkpoint every ~100K steps
        save_path="models/checkpoints",
        name_prefix="ppo_intercept",
    )

    # --- Train! ---
    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60 + "\n")
    start = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Training completed in {elapsed / 60:.1f} minutes")

    # --- Save final model + normalizer ---
    model.save("models/interception_ppo_final")
    train_env.save("models/vecnormalize.pkl")
    print("[SAVED] Final model → models/interception_ppo_final.zip")
    print("[SAVED] Normalizer  → models/vecnormalize.pkl")
    print("[SAVED] Best model  → models/best_model/best_model.zip")


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------
def evaluate(num_episodes=20, use_best=True):
    env = DummyVecEnv([make_env(999)])
    env = VecNormalize.load("models/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    if use_best and os.path.exists("models/best_model/best_model.zip"):
        print("\n[EVAL] Loading BEST model from training")
        model = PPO.load("models/best_model/best_model", env=env)
    else:
        print("\n[EVAL] Loading FINAL model")
        model = PPO.load("models/interception_ppo_final", env=env)

    successes = 0
    total_reward = 0.0
    total_steps = 0

    print(f"\nRunning {num_episodes} evaluation episodes...\n")

    for ep in range(num_episodes):
        obs = env.reset()
        done = [False]
        ep_reward = 0.0
        ep_steps = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

        episode_info = info[0]
        success = episode_info.get("success", False)
        successes += int(success)
        total_reward += ep_reward
        total_steps += ep_steps

        status = "✓ CAUGHT" if success else "✗ MISSED"
        print(f"  Episode {ep + 1:2d}: {status} | reward={ep_reward:7.1f} | steps={ep_steps}")

    print("\n" + "=" * 50)
    print(f"  SUCCESS RATE: {successes}/{num_episodes} = {successes / num_episodes:.0%}")
    print(f"  AVG REWARD:   {total_reward / num_episodes:.1f}")
    print(f"  AVG STEPS:    {total_steps / num_episodes:.1f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train(total_timesteps=20_000_000)    # ~1.5-2 hrs on GPU (overnight run)
    evaluate(num_episodes=20)