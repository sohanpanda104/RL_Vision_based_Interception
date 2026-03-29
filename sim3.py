"""
Stage 3 - Partial Observability: Occlusion Tunnel (Boss Level)
==============================================================
Same randomized conveyor as Stage 2, but now a "tunnel" hides
the object in the middle section of the table.

When the block enters the tunnel:
  - Its position and velocity are MASKED (set to 0) in observations
  - The AI goes "blind" and must predict where the block will exit

This breaks the Markov assumption. We solve it with RecurrentPPO
(LSTM memory) from sb3-contrib.

Optimized for speed: shared LSTM, smaller hidden size, shorter BPTT.
Runs on CPU (faster than GPU for small MLP+LSTM architectures).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
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
class ConveyorOcclusionEnv(gym.Env):
    """
    2D interception environment (Stage 3 - Occlusion Tunnel).

    Same as Stage 2 (domain randomization) but with an occlusion zone
    in the middle of the table. When the object enters this zone, its
    position/velocity are replaced with zeros in the observation.

    The AI must use LSTM memory to:
    1. Remember the object's speed before it entered the tunnel
    2. Predict when and where the object will exit
    3. Position the hand at the exit point in time

    Observation (14-dim):
        hand_x, hand_y,            - hand position
        obj_x, obj_y,              - object position (0 when occluded!)
        obj_vx, obj_vy,            - object velocity (0 when occluded!)
        rel_x, rel_y,              - relative vector  (0 when occluded!)
        distance,                  - scalar distance   (0 when occluded!)
        future_rel_x, future_rel_y - intercept offset  (0 when occluded!)
        mass_factor,               - normalized mass
        belt_speed_norm,           - normalized belt speed
        is_occluded                - 1.0 if object is hidden, 0.0 if visible
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # --- Workspace ---
        self.x_min, self.x_max = -1.0, 1.0
        self.y_min, self.y_max = -1.0, 1.0

        # --- Hand dynamics ---
        self.hand_speed = 0.14

        # --- Object dynamics (randomized, same as Stage 2) ---
        self.belt_speed_min = 0.015
        self.belt_speed_max = 0.050
        self.mass_min = 0.7
        self.mass_max = 1.5
        self.y_drift_max = 0.012

        # --- Occlusion Tunnel ---
        # The tunnel sits in the middle of the table
        # Object enters from the right, exits on the left
        self.tunnel_x_start = 0.30    # right edge of tunnel
        self.tunnel_x_end = -0.15     # left edge of tunnel (object exits here)
        # Tunnel width = 0.45 units. At speed 0.03, takes ~15 steps to cross.

        # --- Success thresholds ---
        self.grasp_radius = 0.10
        self.success_radius = 0.08

        # --- Episode settings ---
        self.max_steps = 350          # slightly more time for tunnel prediction

        # --- Per-episode randomized values ---
        self.current_belt_speed = 0.0
        self.current_mass = 1.0
        self.current_y_drift = 0.0

        # --- Tracking ---
        self.prev_dist = None
        self.prev_hand_pos = None
        self.is_occluded = False
        self.last_visible_vel = None   # remember velocity before occlusion

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 14-dim observation (11 base + mass + belt_speed + is_occluded)
        obs_low = np.array([
            self.x_min, self.y_min,     # hand
            self.x_min, self.y_min,     # object (or 0 when occluded)
            -1.0, -1.0,                 # velocity (or 0 when occluded)
            -3.0, -3.0,                 # relative (or 0 when occluded)
            0.0,                        # distance (or 0 when occluded)
            -3.0, -3.0,                 # future intercept (or 0 when occluded)
            0.0,                        # mass_factor
            0.0,                        # belt_speed (normalized)
            0.0,                        # is_occluded flag
        ], dtype=np.float32)

        obs_high = np.array([
            self.x_max, self.y_max,
            self.x_max, self.y_max,
            1.0, 1.0,
            3.0, 3.0,
            4.0,
            3.0, 3.0,
            1.0,
            1.0,                        # belt_speed (normalized)
            1.0,                        # is_occluded flag
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

    def _is_in_tunnel(self):
        """Check if the object is inside the occlusion tunnel."""
        return self.tunnel_x_end <= self.obj_pos[0] <= self.tunnel_x_start

    def _get_obs(self):
        self.is_occluded = self._is_in_tunnel()

        mass_norm = (self.current_mass - self.mass_min) / (self.mass_max - self.mass_min)
        speed_norm = (self.current_belt_speed - self.belt_speed_min) / (self.belt_speed_max - self.belt_speed_min)

        if self.is_occluded:
            # ============================================================
            #  THE BLINDNESS: Object info is completely masked!
            #  The AI only knows its own hand position and that it's blind.
            # ============================================================
            return np.array([
                self.hand_pos[0], self.hand_pos[1],   # hand (still visible)
                0.0, 0.0,                              # obj position: HIDDEN
                0.0, 0.0,                              # obj velocity: HIDDEN
                0.0, 0.0,                              # relative: HIDDEN
                0.0,                                   # distance: HIDDEN
                0.0, 0.0,                              # future intercept: HIDDEN
                mass_norm,                             # mass (known from before)
                speed_norm,                            # belt speed (known from before)
                1.0,                                   # is_occluded = YES
            ], dtype=np.float32)
        else:
            # Normal visible observation (same as Stage 2)
            rel = self.obj_pos - self.hand_pos
            dist = np.linalg.norm(rel)
            future_obj = self.obj_pos + self.obj_vel * 10
            future_rel = future_obj - self.hand_pos

            # Remember velocity while visible (for internal tracking)
            self.last_visible_vel = self.obj_vel.copy()

            return np.array([
                self.hand_pos[0], self.hand_pos[1],
                self.obj_pos[0], self.obj_pos[1],
                self.obj_vel[0], self.obj_vel[1],
                rel[0], rel[1],
                dist,
                future_rel[0], future_rel[1],
                mass_norm,
                speed_norm,                            # belt speed
                0.0,                                   # is_occluded = NO
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
        self.is_occluded = False
        self.last_visible_vel = None

        # Domain randomization (same as Stage 2)
        self.current_belt_speed = self.np_random.uniform(
            self.belt_speed_min, self.belt_speed_max
        )
        self.current_mass = self.np_random.uniform(
            self.mass_min, self.mass_max
        )
        self.current_y_drift = self.np_random.uniform(
            -self.y_drift_max, self.y_drift_max
        )

        # Hand starts left-center (wider range)
        self.hand_pos = np.array([
            self.np_random.uniform(-0.5, 0.1),
            self.np_random.uniform(-0.4, 0.4),
        ], dtype=np.float32)

        # Object starts on the right edge (BEFORE the tunnel)
        self.obj_pos = np.array([
            self.np_random.uniform(0.55, 0.95),
            self.np_random.uniform(-0.3, 0.3),
        ], dtype=np.float32)

        # Velocity
        vx = -self.current_belt_speed
        vy = self.current_y_drift
        self.obj_vel = np.array([vx, vy], dtype=np.float32)

        # Tracking
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

        # --- Move object (still moves even when hidden!) ---
        self.obj_pos = self._clip(self.obj_pos + self.obj_vel)

        # --- Compute real distances (used for reward, even if AI can't see) ---
        rel = self.obj_pos - self.hand_pos
        dist = float(np.linalg.norm(rel))

        # Is the object currently hidden?
        currently_occluded = self._is_in_tunnel()

        # ===================================================================
        #  REWARD FUNCTION
        # ===================================================================
        reward = 0.0

        # 1) Time penalty
        reward -= 0.3

        # 2) Approach reward (ONLY when object is visible)
        #    When occluded, we can't reward approach because the AI shouldn't
        #    "know" where the object is. But we give a small reward for
        #    moving toward the tunnel exit (anticipatory positioning).
        dist_improvement = self.prev_dist - dist

        if not currently_occluded:
            # Object visible — normal approach reward
            reward += dist_improvement * 50.0
        else:
            # Object hidden — reward moving toward tunnel exit zone
            exit_x = self.tunnel_x_end - 0.05  # slightly left of exit
            exit_dist = abs(self.hand_pos[0] - exit_x)
            if exit_dist < 0.3:
                reward += 2.0 * (0.3 - exit_dist)  # reward being near exit

        # 2b) Urgency bonus (only when visible and near edge)
        if not currently_occluded and self.obj_pos[0] < 0.0 and dist > self.success_radius:
            urgency = max(0.0, 1.0 - (self.obj_pos[0] + 1.0))
            reward += dist_improvement * 30.0 * urgency

        # 3) Proximity bonus (only when visible)
        if not currently_occluded and dist < 0.3:
            reward += 2.0 * (0.3 - dist)

        # 4) Velocity matching (only when visible and close)
        if not currently_occluded and dist < 0.2:
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

        # 6) SUCCESS
        terminated = False
        truncated = False
        info = {
            "success": False,
            "mass": self.current_mass,
            "belt_speed": self.current_belt_speed,
            "caught_in_tunnel": False,
            "caught_after_tunnel": False,
        }

        adjusted_radius = self.success_radius / (0.5 + 0.5 * self.current_mass)

        if gripper_close and dist < adjusted_radius:
            reward += 500.0
            terminated = True
            info["success"] = True
            # Track WHERE the catch happened
            if currently_occluded:
                info["caught_in_tunnel"] = True
            elif self.obj_pos[0] < self.tunnel_x_end:
                info["caught_after_tunnel"] = True

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
        occ = "🔲 HIDDEN" if self._is_in_tunnel() else "👁️ VISIBLE"
        print(
            f"Step {self.steps:3d} | "
            f"hand=({self.hand_pos[0]:+.2f},{self.hand_pos[1]:+.2f}) | "
            f"obj=({self.obj_pos[0]:+.2f},{self.obj_pos[1]:+.2f}) | "
            f"dist={np.linalg.norm(self.obj_pos - self.hand_pos):.3f} | "
            f"{occ}"
        )


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------
def make_env(rank, seed=0):
    def _init():
        env = ConveyorOcclusionEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def get_device():
        print("[INFO] Running on CPU as requested (higher FPS for this architecture)")
        return "cpu"


# ---------------------------------------------------------------------------
#  Train
# ---------------------------------------------------------------------------
def train(total_timesteps=10_000_000):
    os.makedirs("models_s3", exist_ok=True)
    os.makedirs("models_s3/logs", exist_ok=True)
    os.makedirs("models_s3/checkpoints", exist_ok=True)

    device = get_device()
    n_envs = 8                         # reduced from 16 (DummyVecEnv is sequential)

    print(f"\n[STAGE 3] Occlusion Tunnel Training (RecurrentPPO + LSTM)")
    print(f"[INFO] Training for {total_timesteps:,} timesteps across {n_envs} envs")
    print(f"[INFO] Tunnel zone: x ∈ [{-0.15:.2f}, {0.30:.2f}]")
    print(f"[INFO] Optimizations: shared LSTM, hidden=64, n_steps=512")

    # --- Training env ---
    # NOTE: RecurrentPPO does NOT support SubprocVecEnv well with LSTM states.
    # We use DummyVecEnv (runs in same process) for stability.
    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
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

    # --- RecurrentPPO Model (LSTM-powered, optimized!) ---
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=512,                   # ↓ from 2048 (shorter BPTT chain = faster)
        batch_size=256,                # ↓ from 512 (match smaller rollout)
        n_epochs=5,                    # ↓ from 10  (fewer gradient passes)
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,                 # ↓ from 0.03 (less exploration needed)
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        policy_kwargs=dict(
            shared_lstm=True,          # ← KEY: one LSTM for actor+critic (halves BPTT)
            enable_critic_lstm=False,  # ← critic uses shared LSTM output, no own LSTM
            lstm_hidden_size=64,       # ↓ from 128 (smaller matrices, still enough)
            n_lstm_layers=1,
            activation_fn=torch.nn.ReLU,
        ),
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models_s3/best_model",
        log_path="models_s3/logs",
        eval_freq=max(25_000 // n_envs, 1),  # ↑ from 10K (less eval overhead)
        n_eval_episodes=20,                   # ↓ from 30 (faster evals)
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path="models_s3/checkpoints",
        name_prefix="rppo_stage3",
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("  STAGE 3 TRAINING STARTED (Occlusion Tunnel + LSTM)")
    print("=" * 60 + "\n")
    start = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Training completed in {elapsed / 60:.1f} minutes")

    model.save("models_s3/interception_rppo_s3_final")
    train_env.save("models_s3/vecnormalize.pkl")
    print("[SAVED] Final model → models_s3/interception_rppo_s3_final.zip")
    print("[SAVED] Normalizer  → models_s3/vecnormalize.pkl")
    print("[SAVED] Best model  → models_s3/best_model/best_model.zip")


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------
def evaluate(num_episodes=30, use_best=True):
    env = DummyVecEnv([make_env(999)])
    env = VecNormalize.load("models_s3/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    if use_best and os.path.exists("models_s3/best_model/best_model.zip"):
        print("\n[EVAL] Loading BEST model from Stage 3 training")
        model = RecurrentPPO.load("models_s3/best_model/best_model", env=env)
    else:
        print("\n[EVAL] Loading FINAL model")
        model = RecurrentPPO.load("models_s3/interception_rppo_s3_final", env=env)

    successes = 0
    total_reward = 0.0
    total_steps = 0
    catches_after_tunnel = 0
    catches_in_tunnel = 0
    catches_before_tunnel = 0

    print(f"\nRunning {num_episodes} evaluation episodes (occluded tunnel)...\n")

    for ep in range(num_episodes):
        obs = env.reset()

        # RecurrentPPO needs LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        done = [False]
        ep_reward = 0.0
        ep_steps = 0

        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, done, info = env.step(action)
            episode_starts = done  # reset LSTM on episode end
            ep_reward += reward[0]
            ep_steps += 1

        episode_info = info[0]
        success = episode_info.get("success", False)
        mass = episode_info.get("mass", 1.0)
        speed = episode_info.get("belt_speed", 0.03)
        successes += int(success)
        total_reward += ep_reward
        total_steps += ep_steps

        if success:
            if episode_info.get("caught_after_tunnel", False):
                catches_after_tunnel += 1
            elif episode_info.get("caught_in_tunnel", False):
                catches_in_tunnel += 1
            else:
                catches_before_tunnel += 1

        status = "✓ CAUGHT" if success else "✗ MISSED"
        where = ""
        if success:
            if episode_info.get("caught_after_tunnel"):
                where = " [after tunnel]"
            elif episode_info.get("caught_in_tunnel"):
                where = " [inside tunnel]"
            else:
                where = " [before tunnel]"

        print(
            f"  Episode {ep + 1:2d}: {status}{where} | "
            f"reward={ep_reward:7.1f} | steps={ep_steps:3d} | "
            f"speed={speed:.3f} | mass={mass:.2f}"
        )

    print("\n" + "=" * 60)
    print(f"  OVERALL SUCCESS RATE:    {successes}/{num_episodes} = {successes / num_episodes:.0%}")
    print(f"  AVG REWARD:              {total_reward / num_episodes:.1f}")
    print(f"  AVG STEPS:               {total_steps / num_episodes:.1f}")
    print(f"  Caught BEFORE tunnel:    {catches_before_tunnel}")
    print(f"  Caught INSIDE tunnel:    {catches_in_tunnel}")
    print(f"  Caught AFTER tunnel:     {catches_after_tunnel}")
    print(f"  MISSED:                  {num_episodes - successes}")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train(total_timesteps=10_000_000)    # ~1.5-2 hrs with optimizations
    evaluate(num_episodes=30)
