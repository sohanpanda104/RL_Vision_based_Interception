import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def maybe_load_progress_csv(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        return data
    except Exception as exc:
        print(f"[WARN] Could not read progress csv: {exc}")
        return None


def plot_eval_metrics(eval_path, output_path, smooth_window):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation log not found: {eval_path}")

    data = np.load(eval_path, allow_pickle=True)
    timesteps = data["timesteps"]
    rewards = data["results"]
    ep_lengths = data["ep_lengths"]

    reward_mean = rewards.mean(axis=1)
    reward_std = rewards.std(axis=1)
    length_mean = ep_lengths.mean(axis=1)

    has_success = "successes" in data.files
    success_mean = None
    success_std = None
    if has_success:
        successes = data["successes"].astype(float)
        success_mean = successes.mean(axis=1)
        success_std = successes.std(axis=1)

    fig, axes = plt.subplots(3 if has_success else 2, 1, figsize=(12, 12), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].plot(timesteps, reward_mean, color="tab:blue", linewidth=2, label="Mean eval reward")
    axes[0].fill_between(
        timesteps,
        reward_mean - reward_std,
        reward_mean + reward_std,
        color="tab:blue",
        alpha=0.2,
        label="Reward std",
    )
    if smooth_window > 1 and len(reward_mean) >= smooth_window:
        smooth_reward = moving_average(reward_mean, smooth_window)
        axes[0].plot(
            timesteps[smooth_window - 1:],
            smooth_reward,
            color="navy",
            linestyle="--",
            linewidth=2,
            label=f"MA({smooth_window})",
        )
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Evaluation Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    row = 1
    if has_success:
        axes[row].plot(timesteps, success_mean, color="tab:green", linewidth=2, label="Success rate")
        axes[row].fill_between(
            timesteps,
            np.maximum(0.0, success_mean - success_std),
            np.minimum(1.0, success_mean + success_std),
            color="tab:green",
            alpha=0.2,
            label="Success std",
        )
        if smooth_window > 1 and len(success_mean) >= smooth_window:
            smooth_success = moving_average(success_mean, smooth_window)
            axes[row].plot(
                timesteps[smooth_window - 1:],
                smooth_success,
                color="darkgreen",
                linestyle="--",
                linewidth=2,
                label=f"MA({smooth_window})",
            )
        axes[row].set_ylabel("Success")
        axes[row].set_ylim(0.0, 1.05)
        axes[row].set_title("Evaluation Success Rate")
        axes[row].grid(True, alpha=0.3)
        axes[row].legend()
        row += 1

    axes[row].plot(timesteps, length_mean, color="tab:orange", linewidth=2, label="Mean episode length")
    if smooth_window > 1 and len(length_mean) >= smooth_window:
        smooth_length = moving_average(length_mean, smooth_window)
        axes[row].plot(
            timesteps[smooth_window - 1:],
            smooth_length,
            color="darkorange",
            linestyle="--",
            linewidth=2,
            label=f"MA({smooth_window})",
        )
    axes[row].set_xlabel("Timesteps")
    axes[row].set_ylabel("Episode Length")
    axes[row].set_title("Evaluation Episode Length")
    axes[row].grid(True, alpha=0.3)
    axes[row].legend()

    fig.suptitle("Stage 2 Training Metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[SAVED] Evaluation plots -> {output_path}")


def plot_training_reward_from_progress(progress_path, output_path):
    data = maybe_load_progress_csv(progress_path)
    if data is None:
        print("[INFO] No progress.csv found, so no true training-reward plot was generated.")
        return

    names = list(data.dtype.names or [])
    if "time/total_timesteps" not in names or "rollout/ep_rew_mean" not in names:
        print("[INFO] progress.csv exists but does not contain rollout/ep_rew_mean.")
        return

    timesteps = data["time/total_timesteps"]
    rewards = data["rollout/ep_rew_mean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps, rewards, color="tab:red", linewidth=2)
    ax.set_title("Training Reward")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[SAVED] Training reward plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot reward and success graphs for Stage 2 training.")
    parser.add_argument(
        "--eval-file",
        default=os.path.join("models_3d_s2", "eval", "evaluations.npz"),
        help="Path to Stable-Baselines3 EvalCallback evaluations.npz",
    )
    parser.add_argument(
        "--progress-file",
        default=os.path.join("models_3d_s2", "progress.csv"),
        help="Optional SB3 progress.csv path for training reward plotting.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("models_3d_s2", "plots"),
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Moving-average window for smoother curves.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_eval_metrics(
        eval_path=args.eval_file,
        output_path=os.path.join(args.out_dir, "eval_reward_success_length.png"),
        smooth_window=args.smooth_window,
    )
    plot_training_reward_from_progress(
        progress_path=args.progress_file,
        output_path=os.path.join(args.out_dir, "training_reward.png"),
    )


if __name__ == "__main__":
    main()
