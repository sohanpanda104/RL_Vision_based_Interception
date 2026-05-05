# Precision Pick-and-Place: Learning Robotic Control through Reinforcement Learning

A reinforcement learning project where a **7-DOF Franka Emika Panda** robot arm learns to track, intercept, and grasp a moving object on a conveyor belt - even when the object temporarily disappears behind a tunnel.

Built with **PyBullet**, **panda-gym**, and **Stable-Baselines3**.

---

## What This Project Does

We train a simulated robot arm to solve a pick-and-place task that gets progressively harder across three stages:

1. **Stage 1 - Constant Velocity**: The block moves at a fixed speed. The robot learns to intercept it, grasp it, and lift it to a target 20 cm above the table.
2. **Stage 2 - Domain Randomization**: Every episode, the belt speed, object mass, and lateral drift are randomized. The robot must generalize instead of memorizing one trajectory.
3. **Stage 3 - Occlusion Tunnel**: A 20 cm opaque tunnel hides the block mid-transit. The robot can no longer see the block and must predict where it will exit using memory (LSTM).

---

## Results

| Stage                          | Algorithm           | Success Rate           | Training Steps |
| :----------------------------- | :------------------ | :--------------------- | :------------- |
| Stage 1 - Constant Velocity    | PPO (MLP)           | **100%** (20/20) | 4M             |
| Stage 2 - Domain Randomization | PPO (MLP)           | **100%** (30/30) | 5M             |
| Stage 3 - Occlusion Tunnel     | RecurrentPPO (LSTM) | In progress            | 6M             |

---

# How It Works

### Observation Space

The robot receives:

- Its **end-effector position** (x, y, z) and joint states
- The **block's position and velocity** (when visible)
- **Normalized belt speed and mass** (Stages 2 & 3)
- An **`is_occluded` flag** (Stage 3) - tells the robot it cannot see the block right now

During occlusion (Stage 3), the block's true position is replaced with a **trajectory prediction** based on the last known speed and direction.

### Action Space

4D continuous control at each timestep:

- **Δx, Δy, Δz** — end-effector velocity
- **Gripper** — open or close

### Reward System

We use a dense reward with multiple components:

- **Approach penalty** — pulls the hand toward the block
- **Hover bonus** — rewards getting within 5 cm of the block
- **Lift bonus** — rewards picking the block up off the table
- **Target bonus** — pulls the lifted block toward the aerial target
- **Success jackpot** — +100 (Stages 1 & 2) or +150 (Stage 3) for completing the task
- **Early-catch penalty** (Stage 3 only) — penalizes catching the block before the tunnel, forcing the robot to actually use its memory

### Why LSTM for Stage 3?

Standard PPO only sees the current frame. When the block disappears into the tunnel, a standard policy instantly forgets it existed. **RecurrentPPO** adds an LSTM layer that maintains a hidden state across timesteps, letting the robot "remember" the block's trajectory during occlusion.

---

# Methodology

We train agents using:

* **Proximal Policy Optimization (PPO)**
* **MLP-based policies** for fully observable settings
* **Recurrent policies (PPO + LSTM/GRU)** for partial observability.

Comparisons will be made between:

* feedforward vs recurrent policies,
* performance across curriculum stages.

---

## Key Hyperparameters

| Parameter        | Stage 1 & 2      | Stage 3              |
| :--------------- | :--------------- | :------------------- |
| Algorithm        | PPO              | RecurrentPPO         |
| Policy           | MultiInputPolicy | MultiInputLstmPolicy |
| γ (discount)    | 0.8              | **0.995**      |
| Learning rate    | 3×10⁻⁴        | **1×10⁻⁴**  |
| n_steps          | 2048             | **512**        |
| Entropy coeff    | 0.009            | **0.03**       |
| LSTM hidden size | N/A              | **128**        |

The jump from **γ = 0.8 to 0.995** was critical for Stage 3. With 0.8, the robot couldn't connect its early actions to a reward that came 150+ steps later. With 0.995, the delayed reward is still meaningful at the start of the episode.

---

## Contributors

- **Anant Jain** - CSAI, Plaksha University
- **Ramam Agarwal** - CSAI, Plaksha University
- **Sandeep L** - CSAI, Plaksha University
- **Sohan Panda** - RAS, Plaksha University

---

## References

1. [panda-gym](https://github.com/qgallouedec/panda-gym) - Gallouédec et al., NeurIPS 2021
2. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Raffin et al., JMLR 2021
3. [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
4. [PyBullet](https://pybullet.org/) - Coumans & Bai, 2016

## How to run 
1. Download run.sh
2. Run the command **chmod +x run.sh**
3. Then run: **./run.sh** command on the terminal
