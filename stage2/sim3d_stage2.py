"""
working_stage2.py — 3D Conveyor Dynamic Pick & Place (Domain Randomization)
========================================================================
Built on top of working_stage1.py (100% success).
Stage 2 randomizes belt speed, object mass, and Y-drift every episode
so the agent learns to generalize instead of memorizing one pattern.

Action Space:  4D continuous [dx, dy, dz, gripper]
Observation:   Robot EE + Object (pos + vel) + Target + [speed_norm, mass_norm]
Reward:        Same Dense Shaping as Stage 1 (Approach -> Hover -> Lift -> Target)
"""

import os, sys, json, argparse
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONUNBUFFERED"] = "1"

from stable_baselines3 import PPO
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
# DOMAIN RANDOMIZATION RANGES
# ═══════════════════════════════════════════════════════════════

SPEED_MIN, SPEED_MAX = 0.004, 0.015      # Belt speed range
MASS_MIN,  MASS_MAX  = 0.3,  1.2         # Object mass range (kg)
Y_DRIFT_MAX          = 0.005             # Max sideways drift per step

MODEL_DIR = "models_3d_s2"
DEFAULT_FINAL_MODEL = os.path.join(MODEL_DIR, "panda_s2_final.zip")
DEFAULT_VECNORMALIZE = os.path.join(MODEL_DIR, "vecnormalize.pkl")


# ═══════════════════════════════════════════════════════════════
# CUSTOM CALLBACK: Track & Plot Success Rate
# ═══════════════════════════════════════════════════════════════

class SuccessRatePlotCallback(BaseCallback):
    """
    Logs the eval success rate after each evaluation round
    and saves a plot to MODEL_DIR/success_rate_plot.png
    """
    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.timesteps = []
        self.success_rates = []
        self.log_path = os.path.join(save_dir, "success_rate_log.json")

    def _on_step(self):
        # Check if there's a new eval success_rate logged
        if len(self.model.ep_info_buffer) > 0:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "is_success" in info:
                    pass  # We read from the logger instead

        # Read from the EvalCallback's logged data
        if self.logger is not None:
            try:
                sr = self.logger.name_to_value.get("eval/success_rate")
                if sr is not None:
                    ts = self.num_timesteps
                    # Avoid duplicate entries for same timestep
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
        ax.set_title("Stage 2 - Evaluation Success Rate Over Training", fontsize=14)
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
# TASK: Domain-Randomized Conveyor Pick & Place
# ═══════════════════════════════════════════════════════════════

class RandomizedConveyorTask(Task):
    """
    Same as Stage 1, but every episode randomizes:
      - belt_speed  (how fast the block slides)
      - obj_mass    (how heavy the block is)
      - y_drift     (slight sideways wobble on the belt)
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

        # These get randomized every reset
        self.belt_speed = 0.008
        self.obj_mass = 0.5
        self.y_drift = 0.0

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

        # Normalize speed and mass so the network sees values in [0, 1]
        speed_norm = (self.belt_speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN + 1e-8)
        mass_norm  = (self.obj_mass - MASS_MIN) / (MASS_MAX - MASS_MIN + 1e-8)

        # 6 (pos+vel) + 2 (speed, mass) = 8 dims from Task
        return np.concatenate([obj_pos, obj_vel, [speed_norm, mass_norm]])

    def get_achieved_goal(self):
        return np.array(self.sim.get_base_position("object"))

    def reset(self):
        # ─── DOMAIN RANDOMIZATION ───
        self.belt_speed = self.np_random.uniform(SPEED_MIN, SPEED_MAX)
        self.obj_mass   = self.np_random.uniform(MASS_MIN, MASS_MAX)
        self.y_drift    = self.np_random.uniform(-Y_DRIFT_MAX, Y_DRIFT_MAX)

        # Apply the new mass directly via PyBullet (panda_gym wrapper doesn't have this)
        p.changeDynamics(self.sim._bodies_idx["object"], -1, mass=self.obj_mass)

        # Randomize starting Y position (same as Stage 1)
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
            # Move along X (main belt) + slight Y drift
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
# ENVIRONMENT: Same reward structure as Stage 1
# ═══════════════════════════════════════════════════════════════

class PandaRandomizedEnv(RobotTaskEnv):
    def __init__(self, render_mode="rgb_array"):
        sim = PyBullet(render_mode=render_mode, n_substeps=20)
        robot = Panda(sim, block_gripper=False,
                      base_position=np.array([-0.6, 0.0, 0.0]),
                      control_type="ee")
        task = RandomizedConveyorTask(sim, get_ee_position=robot.get_ee_position)

        super().__init__(robot, task,
                         render_distance=1.0, render_yaw=90,
                         render_pitch=-40,
                         render_target_position=np.array([-0.1, 0.0, 0.0]))
        self._max_episode_steps = 250
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.task.step_task()
        self._step_count += 1

        obs_dict, _, terminated, truncated, info = super().step(action)

        # ==========================================
        # SAME DENSE REWARD AS STAGE 1 (proven to work)
        # ==========================================
        ee_pos     = self.robot.get_ee_position()
        obj_pos    = obs_dict["achieved_goal"]
        target_pos = obs_dict["desired_goal"]

        d_ee_obj     = np.linalg.norm(ee_pos - obj_pos)
        d_obj_target = np.linalg.norm(obj_pos - target_pos)

        reward = 0.0

        # 1. Approach Breadcrumb (Pull hand to block)
        reward -= d_ee_obj * 5.0

        # 2. Gripper & Lift Logic
        if d_ee_obj < 0.05:
            reward += 5.0  # Hover bonus

            # If the block is lifted into the air
            if obj_pos[2] > (self.task.object_size / 2 + 0.02):
                reward += 15.0  # Anti-gravity bonus
                reward -= d_obj_target * 10.0  # Pull block to target

        # 3. Ultimate Success
        is_success = d_obj_target < self.task.distance_threshold
        if is_success:
            reward += 100.0
            terminated = True

        info["is_success"] = is_success
        info["belt_speed"] = self.task.belt_speed
        info["obj_mass"]   = self.task.obj_mass

        if self._step_count >= self._max_episode_steps:
            truncated = True

        return obs_dict, float(reward), terminated, truncated, info


# ═══════════════════════════════════════════════════════════════
# ENV FACTORY
# ═══════════════════════════════════════════════════════════════

def make_env(rank, seed=0):
    def _init():
        env = PandaRandomizedEnv(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train(
    total_timesteps=5_000_000,
    n_envs=8,
    eval_freq=25_000,
    eval_episodes=30,
    progress_bar=True,
):
    print("\n" + "=" * 60)
    print("  STAGE 2: Domain Randomization Training")
    print("  Speed: [{:.3f}, {:.3f}]  Mass: [{:.1f}, {:.1f}]".format(
        SPEED_MIN, SPEED_MAX, MASS_MIN, MASS_MAX))
    print("=" * 60)
    sys.stdout.flush()

    EVAL_DIR  = os.path.join(MODEL_DIR, "eval")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ─── Training Environments ───
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ─── Eval Environment ───
    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ─── Callbacks ───
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // n_envs),
        save_path=MODEL_DIR,
        name_prefix="panda_s2",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=EVAL_DIR,
        eval_freq=max(1, eval_freq // n_envs),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    plot_cb = SuccessRatePlotCallback(save_dir=MODEL_DIR)

    # ─── Try loading Stage 1 weights for warm start ───
    s1_model_path = "models_3d_s2/panda_pick_final.zip"
    if os.path.exists(s1_model_path):
        print(f"[WARM START] Loading Stage 1 weights from {s1_model_path}")
        model = PPO.load(
            s1_model_path,
            env=train_env,
            custom_objects={
                "learning_rate": 3e-4,
                "ent_coef": 0.009,
                "gamma": 0.8,
            },
        )
    else:
        print("[FRESH START] No Stage 1 weights found, training from scratch")
        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.8,
            clip_range=0.2,
            ent_coef=0.009,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[ckpt_cb, eval_cb, plot_cb],
        progress_bar=progress_bar,
    )

    model.save(os.path.join(MODEL_DIR, "panda_s2_final"))
    train_env.save(DEFAULT_VECNORMALIZE)
    print(f"\nTraining complete! Model saved to {MODEL_DIR}/")
    print(f"Success rate plot saved to {MODEL_DIR}/success_rate_plot.png")


# ═══════════════════════════════════════════════════════════════
# EVALUATION (GUI + Video Recording)
# ═══════════════════════════════════════════════════════════════

def evaluate_and_record(
    num_episodes=5,
    model_path=DEFAULT_FINAL_MODEL,
    vecnormalize_path=DEFAULT_VECNORMALIZE,
):
    print("\nEvaluating Stage 2 (Domain Randomization)...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Pass --model to evaluate a specific checkpoint zip."
        )
    if not os.path.exists(vecnormalize_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnormalize_path}\n"
            "Train first, or pass --vecnorm to the matching stats file."
        )

    env = PandaRandomizedEnv(render_mode="rgb_array")

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Video recording
    video_folder = "models_3d_s2/videos/"
    os.makedirs(video_folder, exist_ok=True)

    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=env._max_episode_steps * num_episodes,
        name_prefix="panda_s2_randomized",
    )

    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    successes = 0
    ep_count = 0

    for _ in range(env._max_episode_steps * num_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)

        if dones[0]:
            ep_count += 1
            success = infos[0].get("is_success", False)
            speed = infos[0].get("belt_speed", 0)
            mass = infos[0].get("obj_mass", 0)
            successes += int(success)
            status = "✓ LIFTED" if success else "✗ MISSED"
            print(f"  Episode {ep_count:2d}/{num_episodes} | {status} | "
                  f"speed={speed:.4f}, mass={mass:.2f}kg")

    vec_env.close()
    print(f"\nFinal Success Rate: {successes}/{ep_count} "
          f"({successes/max(ep_count,1)*100:.1f}%)")
    print(f"[SAVED] Video saved to: {video_folder}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the Stage 2 Panda conveyor RL setup."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a new model.")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Total PPO timesteps. Use a small value like 10000 for a smoke test.",
    )
    train_parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments to launch during training.",
    )
    train_parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N total timesteps. Lower this for automated smoke runs.",
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Number of episodes per evaluation round.",
    )
    train_parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm/rich progress bar for non-interactive runs.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model.")
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to record.",
    )
    eval_parser.add_argument(
        "--model",
        default=DEFAULT_FINAL_MODEL,
        help="Path to a trained PPO model or checkpoint zip.",
    )
    eval_parser.add_argument(
        "--vecnorm",
        default=DEFAULT_VECNORMALIZE,
        help="Path to the VecNormalize statistics file.",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "eval":
        evaluate_and_record(
            num_episodes=args.episodes,
            model_path=args.model,
            vecnormalize_path=args.vecnorm,
        )
    else:
        train(
            total_timesteps=getattr(args, "timesteps", 5_000_000),
            n_envs=getattr(args, "n_envs", 8),
            eval_freq=getattr(args, "eval_freq", 25_000),
            eval_episodes=getattr(args, "eval_episodes", 30),
            progress_bar=not getattr(args, "no_progress_bar", False),
        )
