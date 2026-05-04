"""
conveyor_stage2_v2.py — Dynamic Mass & Velocity (FIXED)
========================================================
Root cause of "100% train, 0% eval, not grasping":
  1. get_obs() added speed_obs dim → obs space changed from original
     The policy learned with 7D task obs. Original had 6D. This caused
     the policy to fit to a wrong obs → memorized positions not grasping.
  2. Phase 0 had fixed speed=0.008, fixed spawn x=0.3 → agent memorized
     "go to x=0.3, pretend grasp" without actually gripping.
  3. Eval VecNormalize loaded different obs stats than training produced.

FIXES:
  - get_obs() IDENTICAL to original 100% code (6D: obj_pos + obj_vel)
  - Speed/mass variation starts from episode 1 (no memorization phase)
  - Curriculum still exists but Phase 0 has SMALL variation, not zero
  - Eval uses same vecnorm loading pattern that works
  - Model saved from EvalCallback (best model, not last model)
"""

import os, sys
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTHONUNBUFFERED"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder

import panda_gym
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


# ══════════════════════════════════════════════════════════════
# CURRICULUM — starts with small variation immediately
# No "fixed" phase 0 to prevent memorization
# ══════════════════════════════════════════════════════════════
CURRICULUM = [
    # (env_steps,  mass_range,    speed_range)
    (0,            (0.4, 0.6),   (0.006, 0.010)),  # small variation from start
    (500_000,    (0.3, 0.8),   (0.005, 0.011)),  # widen
    (1_500_000,    (0.25, 0.9),  (0.004, 0.013)),  # wider
    (3_000_000,    (0.2,  1.0),  (0.003, 0.014)),  # full range
]


# ══════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════

class SuccessRateCallback(BaseCallback):
    def __init__(self, check_freq=10_000, save_path=".", verbose=1):
        super().__init__(verbose)
        self.check_freq       = check_freq
        self.save_path        = save_path
        self.success_buffer   = []
        self.success_rates    = []
        self.timesteps_log    = []
        self.max_success_rate = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i in range(len(dones)):
            if dones[i] and "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))
        if self.n_calls % self.check_freq == 0 and len(self.success_buffer) > 0:
            rate = float(np.mean(self.success_buffer))
            self.success_rates.append(rate)
            self.timesteps_log.append(self.num_timesteps)
            if rate > self.max_success_rate:
                self.max_success_rate = rate
            if self.verbose:
                print(f"\nStep: {self.num_timesteps} | "
                      f"success={rate:.3f} | max={self.max_success_rate:.3f}")
            self.success_buffer = []
        return True

    def _on_training_end(self):
        if not self.success_rates:
            return
        os.makedirs(self.save_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.timesteps_log, self.success_rates,
                color="steelblue", linewidth=1.5, label="Success rate")
        if len(self.success_rates) >= 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(self.success_rates, kernel, mode="valid")
            ax.plot(self.timesteps_log[9:], smooth,
                    color="darkorange", linewidth=2.5, label="10-pt avg")
        for steps, _, _ in CURRICULUM[1:]:
            ax.axvline(steps, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Success Rate")
        ax.set_title("Dynamic Mass & Speed Training")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(self.save_path, "success_rate_dynamic.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n[SAVED] Plot → {path}")


class EntropyDecayCallback(BaseCallback):
    def __init__(self, start_val=0.009, end_val=0.001,
                 decay_start_frac=0.10, decay_end_frac=0.70, verbose=0):
        super().__init__(verbose)
        self.start_val        = start_val
        self.end_val          = end_val
        self.decay_start_frac = decay_start_frac
        self.decay_end_frac   = decay_end_frac

    def _on_step(self) -> bool:
        frac = self.num_timesteps / self.locals["total_timesteps"]
        if frac < self.decay_start_frac:
            coef = self.start_val
        elif frac > self.decay_end_frac:
            coef = self.end_val
        else:
            t = (frac - self.decay_start_frac) / (self.decay_end_frac - self.decay_start_frac)
            coef = self.start_val + t * (self.end_val - self.start_val)
        self.model.ent_coef = float(coef)
        return True


# ══════════════════════════════════════════════════════════════
# TASK
# KEY FIX: get_obs() is IDENTICAL to original 100% working code
# No extra speed_obs dimension — keeps obs space compatible
# Mass/speed stored in task but NOT added to observation
# ══════════════════════════════════════════════════════════════

class DynamicConveyorTask(Task):

    def __init__(self, sim, get_ee_position,
                 mass_range=(0.4, 0.6),
                 speed_range=(0.006, 0.010)):
        super().__init__(sim)
        self.distance_threshold = 0.05
        self.get_ee_position    = get_ee_position
        self.object_size        = 0.04
        self.prev_obj_pos       = None
        self.belt_direction     = np.array([-1.0, 0.0, 0.0])
        self.obj_x_start        = 0.3
        self.obj_y_range        = 0.08
        self.target_pos         = np.array([0.0, 0.0, 0.20])

        self.mass_range         = mass_range
        self.speed_range        = speed_range
        self.current_mass       = 0.5
        self.current_speed      = speed_range[0]

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=self.current_mass,
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
        # ── IDENTICAL TO ORIGINAL 100% CODE ──────────────────
        # 6D: obj_pos(3) + obj_vel(3)
        # Do NOT add speed_obs — it changes obs space and breaks the policy
        obj_pos = np.array(self.sim.get_base_position("object"))
        if self.prev_obj_pos is not None:
            obj_vel = obj_pos - self.prev_obj_pos
        else:
            obj_vel = np.zeros(3)
        return np.concatenate([obj_pos, obj_vel])

    def get_achieved_goal(self):
        return np.array(self.sim.get_base_position("object"))

    def reset(self):
        # Sample new mass and speed every episode
        self.current_mass  = float(self.np_random.uniform(*self.mass_range))
        self.current_speed = float(self.np_random.uniform(*self.speed_range))

        # Apply mass change to PyBullet body
        obj_id = self.sim._bodies_idx["object"]
        p.changeDynamics(obj_id, -1, mass=self.current_mass)

        y_offset  = self.np_random.uniform(-self.obj_y_range, self.obj_y_range)
        obj_start = np.array([self.obj_x_start, y_offset, self.object_size / 2])
        self.sim.set_base_pose("object", obj_start, np.array([0, 0, 0, 1]))
        self.prev_obj_pos = obj_start.copy()
        self.goal = self.target_pos.copy()

    def step_task(self):
        obj_pos = np.array(self.sim.get_base_position("object"))
        self.prev_obj_pos = obj_pos.copy()
        is_lifted = obj_pos[2] > (self.object_size / 2 + 0.01)
        if not is_lifted:
            new_pos    = obj_pos + self.belt_direction * self.current_speed
            new_pos[2] = self.object_size / 2
            self.sim.set_base_pose("object", new_pos, np.array([0, 0, 0, 1]))
            p.resetBaseVelocity(self.sim._bodies_idx["object"], [0, 0, 0], [0, 0, 0])

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        return 0.0

    def is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold


# ══════════════════════════════════════════════════════════════
# ENV
# ══════════════════════════════════════════════════════════════

class PandaDynamicEnv(RobotTaskEnv):

    def __init__(self, render_mode="rgb_array",
                 mass_range=(0.4, 0.6),
                 speed_range=(0.006, 0.010)):
        # Set ALL instance vars before super().__init__() 
        # because parent calls reset() internally
        self._step_count        = 0
        self._max_episode_steps = 250
        self._total_env_steps   = 0
        self._mass_range        = mass_range
        self._speed_range       = speed_range

        sim   = PyBullet(render_mode=render_mode, n_substeps=20)
        robot = Panda(sim, block_gripper=False,
                      base_position=np.array([-0.6, 0.0, 0.0]),
                      control_type="ee")
        task  = DynamicConveyorTask(
            sim, robot.get_ee_position,
            mass_range  = mass_range,
            speed_range = speed_range,
        )
        super().__init__(robot, task,
                         render_distance=1.0, render_yaw=90,
                         render_pitch=-40,
                         render_target_position=np.array([-0.1, 0.0, 0.0]))

    def reset(self, seed=None, options=None):
        self._step_count = 0
        # Apply current ranges to task BEFORE task.reset() samples them
        self.task.mass_range  = self._mass_range
        self.task.speed_range = self._speed_range
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # Update curriculum ranges based on this env's step count
        self._total_env_steps += 1
        for threshold, mass_r, speed_r in reversed(CURRICULUM):
            if self._total_env_steps >= threshold:
                self._mass_range  = mass_r
                self._speed_range = speed_r
                break

        self.task.step_task()
        self._step_count += 1

        obs_dict, _, terminated, truncated, info = super().step(action)
        terminated = False
        truncated  = False

        ee_pos      = self.robot.get_ee_position()
        obj_pos     = np.array(obs_dict["achieved_goal"])
        target_pos  = np.array(obs_dict["desired_goal"])
        d_ee_obj    = float(np.linalg.norm(ee_pos - obj_pos))
        d_obj_target= float(np.linalg.norm(obj_pos - target_pos))

        '''# ── REWARD: identical to your 100% working code ──────
        reward = 0.0
        reward -= d_ee_obj * 5.0
        if d_ee_obj < 0.05:
            reward += 10.0
            if obj_pos[2] > (self.task.object_size / 2 + 0.02):
                reward += 30.0
                reward -= d_obj_target * 20.0
        is_success = d_obj_target < self.task.distance_threshold
        if is_success:
            reward    += 100.0
            terminated = True
            '''
            

        # ── ANTI-REWARD-HACK VERSION ─────────────

        reward = 0.0

        # distances
        d_ee_obj     = float(np.linalg.norm(ee_pos - obj_pos))
        d_obj_target = float(np.linalg.norm(obj_pos - target_pos))

        # time penalty (CRITICAL)
        reward -= 1.0

        # reach
        reward -= d_ee_obj * 5.0

        # target shaping (reduced)
        reward -= d_obj_target * 3.0

        if d_ee_obj < 0.05:
            reward += 10.0

        # grasp
        is_grasped = (d_ee_obj < 0.03) and (obj_pos[2] > self.task.object_size / 2 + 0.005)
        if is_grasped:
            reward += 15.0

        # lift
        is_lifted = obj_pos[2] > (self.task.object_size / 2 + 0.02)
        if is_lifted:
            reward += 20.0

            # reduced shaping
            reward += (1.0 - d_obj_target) * 3.0

        # success
        is_success = d_obj_target < self.task.distance_threshold
        if is_success:
            reward += 300.0
            terminated = True

        # punish timeout
        if self._step_count >= self._max_episode_steps:
            reward -= 50.0

        info["is_success"] = is_success
        info["block_mass"] = round(self.task.current_mass, 3)
        info["belt_speed"] = round(self.task.current_speed, 4)
        info["height"]     = round(float(obj_pos[2]), 4)

        if self._step_count >= self._max_episode_steps:
            truncated = True

        return obs_dict, float(reward), terminated, truncated, info


# ══════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════

def make_env(rank: int, seed: int = 0,
             mass_range=(0.4, 0.6),
             speed_range=(0.006, 0.010)):
    def _init():
        env = PandaDynamicEnv(
            render_mode = "rgb_array",
            mass_range  = mass_range,
            speed_range = speed_range,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════

def train():
    print("\n=== Stage 2: Dynamic Mass & Velocity v2 ===")
    print("Curriculum (small variation from step 1 to prevent memorization):")
    for steps, mass_r, speed_r in CURRICULUM:
        print(f"  {steps:>9,} steps → mass={mass_r}, speed={speed_r}")
    print()

    N_ENVS          = 8
    TOTAL_TIMESTEPS = 4_500_000
    MODEL_DIR       = "models_dynamic_v2"
    os.makedirs(f"{MODEL_DIR}/best", exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/eval", exist_ok=True)

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # Eval env uses same initial ranges as training phase 0
    eval_env = DummyVecEnv([make_env(99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.83,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        tensorboard_log = f"{MODEL_DIR}/tb",
        policy_kwargs   = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    entropy_cb = EntropyDecayCallback(
        start_val=0.009, end_val=0.001,
        decay_start_frac=0.10, decay_end_frac=0.70,
    )
    success_cb = SuccessRateCallback(check_freq=10_000, save_path=MODEL_DIR)
    ckpt_cb    = CheckpointCallback(
        save_freq=100_000 // N_ENVS, save_path=MODEL_DIR,
        name_prefix="panda_dyn", save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = f"{MODEL_DIR}/best",
        log_path             = f"{MODEL_DIR}/eval",
        eval_freq            = 25_000 // N_ENVS,
        n_eval_episodes      = 20,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [entropy_cb, success_cb, ckpt_cb, eval_cb],
        progress_bar    = True,
    )

    model.save(f"{MODEL_DIR}/final_model")
    train_env.save(f"{MODEL_DIR}/vecnorm.pkl")
    print(f"\nDone → {MODEL_DIR}/")


# ══════════════════════════════════════════════════════════════
# EVALUATE
# Correct VecNormalize pattern: load stats onto raw env,
# then freeze before loading model
# ══════════════════════════════════════════════════════════════

def evaluate(
    model_path  = "models_dynamic_v2/best/best_model",
    vecnorm     = "models_dynamic_v2/vecnorm.pkl",
    n_episodes  = 100,
    render      = True,
    mass_range  = (0.2, 1.0),
    speed_range = (0.003, 0.014),
):
    mode = "human" if render else "rgb_array"

    # Create raw env with desired eval ranges
    raw = DummyVecEnv([lambda: PandaDynamicEnv(
        render_mode = mode,
        mass_range  = mass_range,
        speed_range = speed_range,
    )])

    # Load training normalization stats → policy sees same-scale obs as training
    env = VecNormalize.load(vecnorm, raw)
    env.training    = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    successes = 0
    obs = env.reset()
    for ep in range(n_episodes):
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            steps += 1
            if done:
                ok    = info[0].get("is_success", False)
                mass  = info[0].get("block_mass", "?")
                speed = info[0].get("belt_speed", "?")
                h     = info[0].get("height", "?")
                successes += int(ok)
                print(f"  ep {ep+1:2d} | {'SUCCESS' if ok else 'fail':7s} | "
                      f"steps={steps:3d} | mass={mass}kg | speed={speed} | h={h}")
                obs = env.reset()

    print(f"\nSuccess rate: {successes}/{n_episodes} = {successes/n_episodes*100:.1f}%")
    env.close()


# ══════════════════════════════════════════════════════════════
# RECORD VIDEO
# ══════════════════════════════════════════════════════════════

def record_video(
    model_path  = "models_dynamic_v2/best/best_model",
    vecnorm     = "models_dynamic_v2/vecnorm.pkl",
    n_episodes  = 5,
    mass_range  = (0.2, 1.0),
    speed_range = (0.003, 0.014),
):
    vdir = "models_dynamic_v2/videos"
    os.makedirs(vdir, exist_ok=True)

    raw = DummyVecEnv([lambda: PandaDynamicEnv(
        render_mode = "rgb_array",
        mass_range  = mass_range,
        speed_range = speed_range,
    )])
    env = VecNormalize.load(vecnorm, raw)
    env.training    = False
    env.norm_reward = False
    env = VecVideoRecorder(env, vdir,
                           record_video_trigger=lambda s: s == 0,
                           video_length=250 * n_episodes,
                           name_prefix="panda_dynamic_v2")
    model = PPO.load(model_path, env=env)
    obs   = env.reset()
    for _ in range(250 * n_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        if dones[0] and infos[0].get("is_success"):
            print(f"  SUCCESS — mass={infos[0].get('block_mass','?')} "
                  f"speed={infos[0].get('belt_speed','?')}")
    env.close()
    print(f"Video → {vdir}/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmds = {"train": train, "eval": evaluate, "video": record_video}
    cmd  = sys.argv[1] if len(sys.argv) > 1 else "train"
    cmds.get(cmd, lambda: print(
        "Usage: python conveyor_stage2_v2.py [train|eval|video]"
    ))()