"""
Stage 2 - Domain Randomization: Robustness Test
================================================
Same conveyor interception as Stage 1, but now the physics
change EVERY episode:
  - Belt speed:   randomized (0.015 to 0.05)
  - Object mass:  randomized (affects grip force needed)
  - Y-drift:      randomized (object can drift up/down more)
  - Hand start:   wider randomization

This forces the AI to PAY ATTENTION to its current observations
instead of memorizing a single timing pattern.

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
class ConveyorRandomizedEnv(gym.Env):
    """
    2D interception environment (Stage 2 - Domain Randomization).

    Same as Stage 1 but with randomized physics every episode:
    - Belt speed varies from slow to fast
    - Object mass affects how much force is needed to grip
    - Y-drift makes the object path less predictable
    - Hand and object spawn positions are wider

    Observation (13-dim):
        hand_x, hand_y,            - hand position
        obj_x, obj_y,              - object position
        obj_vx, obj_vy,            - object velocity
        rel_x, rel_y,              - relative vector (obj - hand)
        distance,                  - scalar distance
        future_rel_x, future_rel_y - predicted intercept offset
        mass_factor,               - current object mass (normalized)
        belt_speed_norm            - current belt speed (normalized)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # --- Workspace ---
        self.x_min, self.x_max = -1.0, 1.0
        self.y_min, self.y_max = -1.0, 1.0

        # --- Hand dynamics ---
        self.hand_speed = 0.14

        # --- Object dynamics (RANDOMIZED ranges) ---
        self.belt_speed_min = 0.015    # slowest belt
        self.belt_speed_max = 0.050    # fastest belt
        self.mass_min = 0.7            # lightest object
        self.mass_max = 1.5            # heaviest object
        self.y_drift_max = 0.012       # max lateral drift

        # --- Success thresholds ---
        self.grasp_radius = 0.10
        self.success_radius = 0.08

        # --- Episode settings ---
        self.max_steps = 300

        # --- Per-episode randomized values ---
        self.current_belt_speed = 0.0
        self.current_mass = 1.0        # 1.0 = normal
        self.current_y_drift = 0.0

        # --- Tracking ---
        self.prev_dist = None
        self.prev_hand_pos = None

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 13-dim observation (11 from Stage 1 + 2 new: mass, speed)
        obs_low = np.array([
            self.x_min, self.y_min,     # hand
            self.x_min, self.y_min,     # object
            -1.0, -1.0,                 # velocity
            -3.0, -3.0,                 # relative
            0.0,                        # distance
            -3.0, -3.0,                 # future intercept offset
            0.0,                        # mass_factor (normalized)
            0.0,                        # belt_speed (normalized)
        ], dtype=np.float32)

        obs_high = np.array([
            self.x_max, self.y_max,
            self.x_max, self.y_max,
            1.0, 1.0,
            3.0, 3.0,
            4.0,
            3.0, 3.0,
            1.0,                        # mass_factor (normalized)
            1.0,                        # belt_speed (normalized)
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

        future_obj = self.obj_pos + self.obj_vel * 10
        future_rel = future_obj - self.hand_pos

        # Normalize mass and speed to [0, 1] so the AI can read them
        mass_norm = (self.current_mass - self.mass_min) / (self.mass_max - self.mass_min)
        speed_norm = (self.current_belt_speed - self.belt_speed_min) / (self.belt_speed_max - self.belt_speed_min)

        return np.array([
            self.hand_pos[0], self.hand_pos[1],
            self.obj_pos[0], self.obj_pos[1],
            self.obj_vel[0], self.obj_vel[1],
            rel[0], rel[1],
            dist,
            future_rel[0], future_rel[1],
            mass_norm,
            speed_norm,
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

        # ============================================================
        #  DOMAIN RANDOMIZATION — this is what makes Stage 2 different
        # ============================================================

        # Randomize belt speed
        self.current_belt_speed = self.np_random.uniform(
            self.belt_speed_min, self.belt_speed_max
        )

        # Randomize object mass (affects grip success threshold)
        self.current_mass = self.np_random.uniform(
            self.mass_min, self.mass_max
        )

        # Randomize Y-drift (object can drift sideways)
        self.current_y_drift = self.np_random.uniform(
            -self.y_drift_max, self.y_drift_max
        )

        # Wider hand spawn range than Stage 1
        self.hand_pos = np.array([
            self.np_random.uniform(-0.5, 0.1),
            self.np_random.uniform(-0.4, 0.4),
        ], dtype=np.float32)

        # Object starts on the right edge (wider range)
        self.obj_pos = np.array([
            self.np_random.uniform(0.55, 0.95),
            self.np_random.uniform(-0.4, 0.4),
        ], dtype=np.float32)

        # Velocity: randomized speed + randomized drift
        vx = -self.current_belt_speed
        vy = self.current_y_drift
        self.obj_vel = np.array([vx, vy], dtype=np.float32)

        # Initialize tracking
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
            move_vec = move_vec / move_norm
        self.hand_pos = self._clip(self.hand_pos + move_vec * self.hand_speed)

        # --- Move object ---
        self.obj_pos = self._clip(self.obj_pos + self.obj_vel)

        # --- Compute distances ---
        rel = self.obj_pos - self.hand_pos
        dist = float(np.linalg.norm(rel))

        # ===================================================================
        #  REWARD FUNCTION (same structure as Stage 1)
        # ===================================================================
        reward = 0.0

        # 1) Time penalty
        reward -= 0.3

        # 2) Approach reward
        dist_improvement = self.prev_dist - dist
        reward += dist_improvement * 50.0

        # 2b) Urgency bonus when object is near left edge
        if self.obj_pos[0] < 0.0 and dist > self.success_radius:
            urgency = max(0.0, 1.0 - (self.obj_pos[0] + 1.0))
            reward += dist_improvement * 30.0 * urgency

        # 3) Proximity bonus
        if dist < 0.3:
            reward += 2.0 * (0.3 - dist)

        # 4) Velocity matching bonus
        if dist < 0.2:
            hand_vel = self.hand_pos - self.prev_hand_pos
            hand_vel_norm = np.linalg.norm(hand_vel)
            obj_vel_norm = np.linalg.norm(self.obj_vel)
            if hand_vel_norm > 1e-6 and obj_vel_norm > 1e-6:
                cos_sim = np.dot(hand_vel, self.obj_vel) / (hand_vel_norm * obj_vel_norm)
                reward += 1.0 * max(0.0, cos_sim)

        # 5) Gripper rewards
        gripper_close = grip > 0.5

        if gripper_close and dist < self.grasp_radius:
            reward += 15.0
        elif gripper_close and dist > 0.25:
            reward -= 3.0

        # 6) SUCCESS — mass-adjusted success radius
        #    Heavier objects are slightly harder to grasp (tighter threshold)
        terminated = False
        truncated = False
        info = {
            "success": False,
            "mass": self.current_mass,
            "belt_speed": self.current_belt_speed,
        }

        # Heavier objects need a slightly closer grasp
        adjusted_radius = self.success_radius / (0.5 + 0.5 * self.current_mass)

        if gripper_close and dist < adjusted_radius:
            reward += 500.0
            terminated = True
            info["success"] = True

        # 7) Object escapes
        if self.obj_pos[0] <= self.x_min + 0.01:
            reward -= 50.0
            terminated = True
            info["missed"] = True

        # 8) Max steps
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
            f"dist={np.linalg.norm(self.obj_pos - self.hand_pos):.3f} | "
            f"speed={self.current_belt_speed:.3f} | "
            f"mass={self.current_mass:.2f}"
        )


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------
def make_env(rank, seed=0):
    def _init():
        env = ConveyorRandomizedEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


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
def train(total_timesteps=20_000_000):
    os.makedirs("models_s2", exist_ok=True)
    os.makedirs("models_s2/logs", exist_ok=True)
    os.makedirs("models_s2/checkpoints", exist_ok=True)

    device = get_device()
    n_envs = 16

    print(f"\n[STAGE 2] Domain Randomization Training")
    print(f"[INFO] Training for {total_timesteps:,} timesteps across {n_envs} envs")

    # --- Training env ---
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # --- Eval env ---
    eval_env = DummyVecEnv([make_env(100)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    eval_env.training = False

    # --- PPO Model ---
    # Slightly larger network for domain randomization
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.025,                # slightly more exploration for varied physics
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models_s2/best_model",
        log_path="models_s2/logs",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=30,            # more eval episodes for randomized env
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path="models_s2/checkpoints",
        name_prefix="ppo_stage2",
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("  STAGE 2 TRAINING STARTED (Domain Randomization)")
    print("=" * 60 + "\n")
    start = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Training completed in {elapsed / 60:.1f} minutes")

    model.save("models_s2/interception_ppo_s2_final")
    train_env.save("models_s2/vecnormalize.pkl")
    print("[SAVED] Final model → models_s2/interception_ppo_s2_final.zip")
    print("[SAVED] Normalizer  → models_s2/vecnormalize.pkl")
    print("[SAVED] Best model  → models_s2/best_model/best_model.zip")


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------
def evaluate(num_episodes=30, use_best=True):
    env = DummyVecEnv([make_env(999)])
    env = VecNormalize.load("models_s2/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    if use_best and os.path.exists("models_s2/best_model/best_model.zip"):
        print("\n[EVAL] Loading BEST model from Stage 2 training")
        model = PPO.load("models_s2/best_model/best_model", env=env)
    else:
        print("\n[EVAL] Loading FINAL model")
        model = PPO.load("models_s2/interception_ppo_s2_final", env=env)

    successes = 0
    total_reward = 0.0
    total_steps = 0

    # Track performance by speed category
    fast_success = 0
    fast_total = 0
    slow_success = 0
    slow_total = 0

    print(f"\nRunning {num_episodes} evaluation episodes (randomized physics)...\n")

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
        mass = episode_info.get("mass", 1.0)
        speed = episode_info.get("belt_speed", 0.03)
        successes += int(success)
        total_reward += ep_reward
        total_steps += ep_steps

        # Categorize by speed
        mid_speed = (0.015 + 0.05) / 2
        if speed > mid_speed:
            fast_total += 1
            fast_success += int(success)
        else:
            slow_total += 1
            slow_success += int(success)

        status = "✓ CAUGHT" if success else "✗ MISSED"
        print(
            f"  Episode {ep + 1:2d}: {status} | "
            f"reward={ep_reward:7.1f} | steps={ep_steps:3d} | "
            f"speed={speed:.3f} | mass={mass:.2f}"
        )

    print("\n" + "=" * 60)
    print(f"  OVERALL SUCCESS RATE: {successes}/{num_episodes} = {successes / num_episodes:.0%}")
    print(f"  AVG REWARD:           {total_reward / num_episodes:.1f}")
    print(f"  AVG STEPS:            {total_steps / num_episodes:.1f}")
    print(f"  SLOW BELT (<{mid_speed:.3f}):    {slow_success}/{slow_total} = {slow_success / max(slow_total, 1):.0%}")
    print(f"  FAST BELT (>{mid_speed:.3f}):    {fast_success}/{fast_total} = {fast_success / max(fast_total, 1):.0%}")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train(total_timesteps=20_000_000)    # ~1.5-2 hrs on GPU (overnight)
    evaluate(num_episodes=30)
