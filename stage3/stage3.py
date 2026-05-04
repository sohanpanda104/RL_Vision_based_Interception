"""
working_stage3.py — 3D Conveyor Dynamic Pick & Place (Occlusion Tunnel + LSTM)
========================================================================
Built on top of working_stage2.py (100% success).
Stage 3 adds an occlusion tunnel that hides the block mid-transit.
The agent must use LSTM memory to predict the block's exit position.

Algorithm:     RecurrentPPO (PPO + LSTM memory)
Action Space:  4D continuous [dx, dy, dz, gripper]
Observation:   Robot EE + Object (pos + vel) + Target + [speed_norm, mass_norm, is_occluded]
               (Object pos/vel are ZEROED when inside the tunnel)
Reward:        Same Dense Shaping as Stage 1 & 2 (proven 100% success)
"""

import os, sys, json
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONUNBUFFERED"] = "1"

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder

import panda_gym
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


# ═══════════════════════════════════════════════════════════════
# DOMAIN RANDOMIZATION RANGES (same as Stage 2)
# ═══════════════════════════════════════════════════════════════

SPEED_MIN, SPEED_MAX = 0.004, 0.015
MASS_MIN,  MASS_MAX  = 0.3,  1.2
Y_DRIFT_MAX          = 0.005

# ═══════════════════════════════════════════════════════════════
# TUNNEL BOUNDARIES (on the X-axis, block moves in -X direction)
# ═══════════════════════════════════════════════════════════════
# Block spawns at x=0.3, moves toward -x
# It enters the tunnel at x=0.25, exits at x=-0.05
# Tunnel length = 0.20 in X → hidden for ~13-50 steps depending on speed

TUNNEL_ENTER_X = 0.25    # Block enters tunnel (goes blind)
TUNNEL_EXIT_X  = -0.05   # Block exits tunnel (vision returns)


# ═══════════════════════════════════════════════════════════════
# CUSTOM CALLBACK: Track & Plot Success Rate
# ═══════════════════════════════════════════════════════════════

class SuccessRatePlotCallback(BaseCallback):
    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.timesteps = []
        self.success_rates = []
        self.log_path = os.path.join(save_dir, "success_rate_log.json")

    def _on_step(self):
        if self.logger is not None:
            try:
                sr = self.logger.name_to_value.get("eval/success_rate")
                if sr is not None:
                    ts = self.num_timesteps
                    if len(self.timesteps) == 0 or self.timesteps[-1] != ts:
                        self.timesteps.append(ts)
                        self.success_rates.append(sr)
                        self._save_plot()
                        self._save_log()
            except Exception:
                pass
        return True

    def _save_plot(self):
        if len(self.timesteps) < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timesteps, self.success_rates, "b-o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title("Stage 3 - Occlusion Tunnel: Success Rate Over Training", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="100% target")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "success_rate_plot.png"), dpi=150)
        plt.close(fig)

    def _save_log(self):
        data = {"timesteps": self.timesteps, "success_rates": self.success_rates}
        with open(self.log_path, "w") as f:
            json.dump(data, f)


# ═══════════════════════════════════════════════════════════════
# TASK: Occluded Conveyor Pick & Place
# ═══════════════════════════════════════════════════════════════

class OccludedConveyorTask(Task):
    """
    Same randomized conveyor as Stage 2, but with a tunnel that
    hides the block between TUNNEL_ENTER_X and TUNNEL_EXIT_X.
    When occluded, object pos/vel observations are zeroed out.
    """

    def __init__(self, sim, get_ee_position):
        super().__init__(sim)
        self.distance_threshold = 0.05
        self.get_ee_position = get_ee_position
        self.object_size = 0.04
        self.prev_obj_pos = None

        self.belt_direction = np.array([-1.0, 0.0, 0.0])
        self.obj_x_start = 0.3
        self.obj_y_range = 0.08

        # Target: 20cm in the air
        self.target_pos = np.array([0.0, 0.0, 0.20])

        # These get randomized every reset (same as Stage 2)
        self.belt_speed = 0.008
        self.obj_mass = 0.5
        self.y_drift = 0.0
        self.is_occluded = 0.0

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=self.obj_mass,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.2, 0.2, 1.0]),
        )

        # ─── THE TUNNEL (visual only, ghost=True so it doesn't block physics) ───
        tunnel_length_x = TUNNEL_ENTER_X - TUNNEL_EXIT_X  # 0.20
        tunnel_center_x = (TUNNEL_ENTER_X + TUNNEL_EXIT_X) / 2.0  # 0.05
        self.sim.create_box(
            body_name="tunnel",
            half_extents=np.array([tunnel_length_x / 2, 0.15, 0.04]),
            mass=0.0,
            ghost=True,
            position=np.array([tunnel_center_x, 0.0, 0.06]),
            rgba_color=np.array([0.3, 0.3, 0.3, 0.4]),  # Semi-transparent grey
        )

        self.sim.create_sphere(
            body_name="target",
            radius=0.02, mass=0.0, ghost=True,
            position=self.target_pos,
            rgba_color=np.array([0.1, 0.9, 0.1, 0.4]),
        )

    def get_obs(self):
        obj_pos = np.array(self.sim.get_base_position("object"))

        if self.prev_obj_pos is not None:
            obj_vel = obj_pos - self.prev_obj_pos
        else:
            obj_vel = np.zeros(3)

        # ─── OCCLUSION LOGIC ───
        # If the block is between the tunnel walls AND still on the table, zero it out
        is_on_table = obj_pos[2] < (self.object_size / 2 + 0.02)
        is_in_tunnel_x = TUNNEL_EXIT_X <= obj_pos[0] <= TUNNEL_ENTER_X

        if is_in_tunnel_x and is_on_table:
            self.is_occluded = 1.0
            obs_obj = np.zeros(6)  # Robot goes BLIND
        else:
            self.is_occluded = 0.0
            obs_obj = np.concatenate([obj_pos, obj_vel])

        # Normalize speed and mass (same as Stage 2)
        speed_norm = (self.belt_speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN + 1e-8)
        mass_norm  = (self.obj_mass - MASS_MIN) / (MASS_MAX - MASS_MIN + 1e-8)

        # 6 (pos+vel or zeros) + 3 (speed, mass, occluded) = 9 dims from Task
        return np.concatenate([obs_obj, [speed_norm, mass_norm, self.is_occluded]])

    def get_achieved_goal(self):
        return np.array(self.sim.get_base_position("object"))

    def reset(self):
        # ─── DOMAIN RANDOMIZATION (same as Stage 2) ───
        self.belt_speed = self.np_random.uniform(SPEED_MIN, SPEED_MAX)
        self.obj_mass   = self.np_random.uniform(MASS_MIN, MASS_MAX)
        self.y_drift    = self.np_random.uniform(-Y_DRIFT_MAX, Y_DRIFT_MAX)
        self.is_occluded = 0.0

        # Apply the new mass directly via PyBullet
        p.changeDynamics(self.sim._bodies_idx["object"], -1, mass=self.obj_mass)

        # Randomize starting Y position
        y_offset = self.np_random.uniform(-self.obj_y_range, self.obj_y_range)
        obj_start = np.array([self.obj_x_start, y_offset, self.object_size / 2])

        self.sim.set_base_pose("object", obj_start, np.array([0, 0, 0, 1]))
        p.resetBaseVelocity(self.sim._bodies_idx["object"], [0, 0, 0], [0, 0, 0])
        self.prev_obj_pos = obj_start.copy()

        # Goal is fixed in the air
        self.goal = self.target_pos.copy()

    def step_task(self):
        """Conveyor physics with Y-drift. Turns off when lifted."""
        obj_pos = np.array(self.sim.get_base_position("object"))
        self.prev_obj_pos = obj_pos.copy()

        is_lifted = obj_pos[2] > (self.object_size / 2 + 0.01)

        if not is_lifted:
            new_pos = obj_pos + self.belt_direction * self.belt_speed
            new_pos[1] += self.y_drift
            new_pos[2] = self.object_size / 2
            self.sim.set_base_pose("object", new_pos, np.array([0, 0, 0, 1]))
            p.resetBaseVelocity(self.sim._bodies_idx["object"], [0, 0, 0], [0, 0, 0])

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        return 0.0

    def is_success(self, achieved_goal, desired_goal):
        d_obj_target = np.linalg.norm(achieved_goal - desired_goal)
        return d_obj_target < self.distance_threshold


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT: Same reward as Stage 1 & 2 + catch location tracking
# ═══════════════════════════════════════════════════════════════

class PandaOccludedEnv(RobotTaskEnv):
    def __init__(self, render_mode="rgb_array"):
        sim = PyBullet(render_mode=render_mode, n_substeps=20)
        robot = Panda(sim, block_gripper=False,
                      base_position=np.array([-0.6, 0.0, 0.0]),
                      control_type="ee")
        task = OccludedConveyorTask(sim, get_ee_position=robot.get_ee_position)

        super().__init__(robot, task,
                         render_distance=1.0, render_yaw=90,
                         render_pitch=-40,
                         render_target_position=np.array([-0.1, 0.0, 0.0]))
        self._max_episode_steps = 300  # More time for prediction + grasping
        self._step_count = 0
        self._catch_location = "missed"

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._catch_location = "missed"
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.task.step_task()
        self._step_count += 1

        obs_dict, _, terminated, truncated, info = super().step(action)

        # ==========================================
        # SAME DENSE REWARD AS STAGE 1 & 2 (proven 100%)
        # ==========================================
        ee_pos     = self.robot.get_ee_position()
        obj_pos    = obs_dict["achieved_goal"]
        target_pos = obs_dict["desired_goal"]

        d_ee_obj     = np.linalg.norm(ee_pos - obj_pos)
        d_obj_target = np.linalg.norm(obj_pos - target_pos)

        reward = 0.0

        # 1. Approach Breadcrumb (Pull hand to block)
        reward -= d_ee_obj * 5.3

        # 2. Gripper & Lift Logic
        if d_ee_obj < 0.05:
            reward += 15.0  # Hover bonus

            # If the block is lifted into the air
            if obj_pos[2] > (self.task.object_size / 2 + 0.02):
                reward += 25.0  # Anti-gravity bonus
                reward -= d_obj_target * 10.5  # Pull block to target

        # 3. Ultimate Success
        is_success = d_obj_target < self.task.distance_threshold
        if is_success:
            reward += 300.0
            terminated = True

            # ─── Track WHERE the catch happened ───
            if obj_pos[0] > TUNNEL_ENTER_X:
                self._catch_location = "before_tunnel"
            elif obj_pos[0] >= TUNNEL_EXIT_X:
                self._catch_location = "inside_tunnel"
            else:
                self._catch_location = "after_tunnel"

        info["is_success"]      = is_success
        info["belt_speed"]      = self.task.belt_speed
        info["obj_mass"]        = self.task.obj_mass
        info["catch_location"]  = self._catch_location

        if self._step_count >= self._max_episode_steps:
            truncated = True

        return obs_dict, float(reward), terminated, truncated, info


# ═══════════════════════════════════════════════════════════════
# ENV FACTORY
# ═══════════════════════════════════════════════════════════════

def make_env(rank, seed=0):
    def _init():
        env = PandaOccludedEnv(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train():
    print("\n" + "=" * 60)
    print("  STAGE 3: Occlusion Tunnel + LSTM Training")
    print("  Tunnel X: [{:.2f}, {:.2f}]".format(TUNNEL_EXIT_X, TUNNEL_ENTER_X))
    print("  Speed: [{:.3f}, {:.3f}]  Mass: [{:.1f}, {:.1f}]".format(
        SPEED_MIN, SPEED_MAX, MASS_MIN, MASS_MAX))
    print("=" * 60)
    sys.stdout.flush()

    N_ENVS = 8
    TOTAL_TIMESTEPS = 5_000_000
    MODEL_DIR = "models_3d_s3"
    EVAL_DIR  = os.path.join(MODEL_DIR, "eval")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ─── Training Environments ───
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ─── Eval Environment ───
    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ─── Callbacks ───
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // N_ENVS),
        save_path=MODEL_DIR,
        name_prefix="panda_s3",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=EVAL_DIR,
        eval_freq=max(1, 25_000 // N_ENVS),
        n_eval_episodes=30,
        deterministic=True,
        render=False,
        verbose=1,
    )
    plot_cb = SuccessRatePlotCallback(save_dir=MODEL_DIR)

    # ─── RecurrentPPO with LSTM ───
    policy_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=True,         # Single LSTM shared between actor & critic
        enable_critic_lstm=False,  # Critic uses shared LSTM output
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,              # Shorter rollouts for LSTM (gradient chains)
        batch_size=256,
        n_epochs=5,               # Fewer epochs (LSTM is sensitive to over-training)
        gamma=0.81,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.009,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[ckpt_cb, eval_cb, plot_cb],
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "panda_s3_final"))
    train_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\nTraining complete! Model saved to {MODEL_DIR}/")
    print(f"Success rate plot saved to {MODEL_DIR}/success_rate_plot.png")


# ═══════════════════════════════════════════════════════════════
# EVALUATION (Video Recording + Catch Location Stats)
# ═══════════════════════════════════════════════════════════════

def evaluate_and_record(num_episodes=15):
    print("\nEvaluating Stage 3 (Occlusion Tunnel + LSTM)...")

    env = PandaOccludedEnv(render_mode="rgb_array")

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load("models_3d_s3/vecnormalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Video recording
    video_folder = "models_3d_s3/videos/"
    os.makedirs(video_folder, exist_ok=True)

    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=env._max_episode_steps * num_episodes,
        name_prefix="panda_s3_occlusion",
    )

    model = RecurrentPPO.load("models_3d_s3/panda_s3_final", env=vec_env)

    obs = vec_env.reset()
    successes = 0
    ep_count = 0
    catch_stats = {"before_tunnel": 0, "inside_tunnel": 0, "after_tunnel": 0, "missed": 0}

    # LSTM state management
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for _ in range(env._max_episode_steps * num_episodes):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, _, dones, infos = vec_env.step(action)
        episode_starts = dones

        if dones[0]:
            ep_count += 1
            success = infos[0].get("is_success", False)
            speed = infos[0].get("belt_speed", 0)
            mass = infos[0].get("obj_mass", 0)
            location = infos[0].get("catch_location", "missed")

            successes += int(success)
            catch_stats[location] += 1

            status = "✓ LIFTED" if success else "✗ MISSED"
            print(f"  Episode {ep_count:2d}/{num_episodes} | {status} "
                  f"[{location}] | speed={speed:.4f}, mass={mass:.2f}kg")

    vec_env.close()

    print(f"\n{'='*50}")
    print(f"Final Success Rate: {successes}/{ep_count} "
          f"({successes/max(ep_count,1)*100:.1f}%)")
    print(f"{'='*50}")
    print(f"  Before tunnel: {catch_stats['before_tunnel']}")
    print(f"  Inside tunnel: {catch_stats['inside_tunnel']}")
    print(f"  After tunnel:  {catch_stats['after_tunnel']}")
    print(f"  Missed:        {catch_stats['missed']}")
    print(f"\n[SAVED] Video saved to: {video_folder}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_and_record(15)
    else:
        train()