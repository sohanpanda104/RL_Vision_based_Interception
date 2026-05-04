"""
sim3d_stage2.py — 3D Conveyor Dynamic Pick & Place
========================================================================
Uses panda-gym (PyBullet) Franka Emika Panda 7-DOF arm.
The robot must intercept a moving block, grasp it, and lift it 
to a target coordinate floating in the air.

Action Space:  4D continuous [dx, dy, dz, gripper]
Observation:   Robot EE + Object (pos + vel) + Target
Reward:        Custom Dense Shaping (Approach -> Hover -> Lift -> Target)
"""

import os, time, sys
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

os.environ["PYTHONUNBUFFERED"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

import panda_gym
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from stable_baselines3.common.vec_env import VecVideoRecorder


# ─── Custom Task: Dynamic Conveyor Pick & Place ───────────────

class ConveyorPickAndPlaceTask(Task):
    """
    A block slides across the table.
    The robot must pick it up and move it to a target in the air.
    Achieved Goal: Object Position
    Desired Goal: Target Position (in the air)
    """

    def __init__(self, sim, get_ee_position, belt_speed=0.008):
        super().__init__(sim)
        self.distance_threshold = 0.05
        self.get_ee_position = get_ee_position
        self.belt_speed = belt_speed
        self.object_size = 0.04
        self.prev_obj_pos = None

        self.belt_direction = np.array([-1.0, 0.0, 0.0])
        self.obj_x_start = 0.3     
        self.obj_y_range = 0.08    
        
        # THE NEW GOAL: 20cm in the air, right in the middle of the table
        self.target_pos = np.array([0.0, 0.0, 0.20])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.5,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.2, 0.2, 1.0]),  # Red block
        )
        
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=self.target_pos,
            rgba_color=np.array([0.1, 0.9, 0.1, 0.4]),  # Green ghost sphere in the air
        )

    def get_obs(self):
        obj_pos = np.array(self.sim.get_base_position("object"))
        if self.prev_obj_pos is not None:
            obj_vel = obj_pos - self.prev_obj_pos
        else:
            obj_vel = np.zeros(3)
        return np.concatenate([obj_pos, obj_vel])

    def get_achieved_goal(self):
        # The goal is now tied to the OBJECT, not the hand
        return np.array(self.sim.get_base_position("object"))

    def reset(self):
        y_offset = self.np_random.uniform(-self.obj_y_range, self.obj_y_range)
        obj_start = np.array([self.obj_x_start, y_offset, self.object_size / 2])
        
        self.sim.set_base_pose("object", obj_start, np.array([0, 0, 0, 1]))
        self.prev_obj_pos = obj_start.copy()
        
        # Goal is fixed in the air
        self.goal = self.target_pos.copy()

    def step_task(self):
        """Conveyor physics that turn off when lifted."""
        obj_pos = np.array(self.sim.get_base_position("object"))
        self.prev_obj_pos = obj_pos.copy()

        # LIFT SENSOR: If the block is more than 1cm off the table, it is lifted!
        is_lifted = obj_pos[2] > (self.object_size / 2 + 0.01)

        if not is_lifted:
            # Apply conveyor movement ONLY if it's still on the table
            new_pos = obj_pos + self.belt_direction * self.belt_speed
            new_pos[2] = self.object_size / 2 
            self.sim.set_base_pose("object", new_pos, np.array([0, 0, 0, 1]))
            p.resetBaseVelocity(self.sim._bodies_idx["object"], [0, 0, 0], [0, 0, 0])

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        # We override this and calculate the dense reward in the Env step() below
        return 0.0

    def is_success(self, achieved_goal, desired_goal):
        d_obj_target = np.linalg.norm(achieved_goal - desired_goal)
        return d_obj_target < self.distance_threshold


# ─── Custom Environment: Custom Reward Shaping ─────────────────────────

class PandaPickAndPlaceEnv(RobotTaskEnv):
    """
    Overrides the step function to inject our custom Dense Breadcrumbs.
    """
    def __init__(self, render_mode="rgb_array", belt_speed=0.008):
        sim = PyBullet(render_mode=render_mode, n_substeps=20)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
        task = ConveyorPickAndPlaceTask(sim, get_ee_position=robot.get_ee_position, belt_speed=belt_speed)
        
        super().__init__(robot, task, render_distance=1.0, render_yaw=90, render_pitch=-40, render_target_position=np.array([-0.1, 0.0, 0.0]))
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
        # CUSTOM DENSE REWARD FOR PICK & PLACE
        # ==========================================
        ee_pos = self.robot.get_ee_position()
        obj_pos = obs_dict["achieved_goal"]
        target_pos = obs_dict["desired_goal"]

        d_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_target = np.linalg.norm(obj_pos - target_pos)

        reward = 0.0

        # 1. Approach Breadcrumb (Pull hand to block)
        reward -= d_ee_obj * 5.0

        # 2. Gripper & Lift Logic
        if d_ee_obj < 0.05:
            reward += 5.0  # Hover bonus
            
            # If the block is lifted into the air!
            if obj_pos[2] > (self.task.object_size / 2 + 0.02):
                reward += 15.0  # Massive bonus for fighting gravity
                reward -= d_obj_target * 10.0  # Pull the floating block to the target

        # 3. Ultimate Success
        is_success = d_obj_target < self.task.distance_threshold
        if is_success:
            reward += 100.0
            terminated = True
            
        info["is_success"] = is_success

        if self._step_count >= self._max_episode_steps:
            truncated = True

        return obs_dict, float(reward), terminated, truncated, info


# ─── Environment Factory ─────────────────────────────────────────────

def make_env(rank, seed=0, belt_speed=0.008):
    def _init():
        env = PandaPickAndPlaceEnv(render_mode="rgb_array", belt_speed=belt_speed)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# ─── Training Script ─────────────────────────────────────────────

def train():
    print("\nStarting 3D Dynamic Pick & Place Training...")
    N_ENVS = 8
    TOTAL_TIMESTEPS = 4_000_000  # Needs more time to learn gravity/grasping
    MODEL_DIR = "models_3d_s2"
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.8,
        clip_range=0.2,
        ent_coef=0.009, # Higher exploration needed to find the grasp
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=MODEL_DIR, name_prefix="panda_lift")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[ckpt_cb], progress_bar=True)

    model.save(f"{MODEL_DIR}/panda_pick_final")
    env.save(f"{MODEL_DIR}/vecnormalize.pkl")
    print("Training Complete!")


def evaluate_and_record(num_episodes=3):
    print("\nEvaluating and Recording 3D Pick & Place...")
    
    # CRITICAL: For recording video, render_mode MUST be "rgb_array", not "human"
    env = PandaPickAndPlaceEnv(render_mode="rgb_array")
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load("models_3d_s2/vecnormalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # ==========================================
    # THE VIDEO RECORDER WRAPPER
    # ==========================================
    video_folder = "models_3d_s2/videos/"
    os.makedirs(video_folder, exist_ok=True)
    
    # We will record the episodes. The trigger lambda x: x == 0 means 
    # it starts recording at step 0 (the very beginning).
    vec_env = VecVideoRecorder(
        vec_env, 
        video_folder,
        record_video_trigger=lambda step: step == 0, 
        video_length=env._max_episode_steps * num_episodes, # Record exactly this many episodes
        name_prefix="panda_pick_and_place"
    )

    # Load the trained brain
    model = PPO.load("models_3d_s2/panda_pick_final", env=vec_env)

    obs = vec_env.reset()
    
    # Run the episodes. The VecVideoRecorder captures frames automatically in the background!
    for _ in range(env._max_episode_steps * num_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        
        if dones[0] and infos[0].get("is_success"):
             print("SUCCESS - Block lifted to target!")

    # You MUST close the environment to finalize and save the .mp4 file
    vec_env.close()
    print(f"\n[SAVED] Video successfully saved to: {video_folder}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_and_record(5)
    else:
        train()