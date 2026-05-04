import argparse
import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from sim3d_stage2 import PandaRandomizedEnv, MODEL_DIR, DEFAULT_FINAL_MODEL, DEFAULT_VECNORMALIZE


def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "*.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Smoke-test sim3d_stage2 and save a video."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional PPO model zip. If omitted, the script tries the final model, then the latest checkpoint, then random actions.",
    )
    parser.add_argument(
        "--vecnorm",
        default=DEFAULT_VECNORMALIZE,
        help="Optional VecNormalize stats file.",
    )
    parser.add_argument(
        "--video-dir",
        default=os.path.join(MODEL_DIR, "test_videos"),
        help="Directory where the MP4 video will be saved.",
    )
    return parser


def resolve_model_path(explicit_model):
    if explicit_model:
        return explicit_model
    if os.path.exists(DEFAULT_FINAL_MODEL):
        return DEFAULT_FINAL_MODEL
    return find_latest_checkpoint(MODEL_DIR)


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.video_dir, exist_ok=True)

    env = PandaRandomizedEnv(render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: env])

    use_vecnorm = os.path.exists(args.vecnorm)
    if use_vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"[WARN] VecNormalize file not found: {args.vecnorm}")
        print("[WARN] Continuing without normalization stats.")

    total_steps = env._max_episode_steps * args.episodes
    vec_env = VecVideoRecorder(
        vec_env,
        args.video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=total_steps,
        name_prefix="stage2_test",
    )

    model_path = resolve_model_path(args.model)
    model = None
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Using model: {model_path}")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("[INFO] No model found. Running a random-policy smoke test.")

    obs = vec_env.reset()
    successes = 0
    completed_episodes = 0

    for _ in range(total_steps):
        if model is None:
            action = [vec_env.action_space.sample()]
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, dones, infos = vec_env.step(action)

        if dones[0]:
            completed_episodes += 1
            success = bool(infos[0].get("is_success", False))
            successes += int(success)
            status = "SUCCESS" if success else "MISS"
            print(
                f"Episode {completed_episodes:2d}/{args.episodes} | "
                f"{status} | "
                f"speed={infos[0].get('belt_speed', 0):.4f} | "
                f"mass={infos[0].get('obj_mass', 0):.2f}kg"
            )

    vec_env.close()
    print(
        f"\nDone. Success rate: {successes}/{max(completed_episodes, 1)} "
        f"({successes / max(completed_episodes, 1) * 100:.1f}%)"
    )
    print(f"Video saved under: {args.video_dir}")


if __name__ == "__main__":
    main()
