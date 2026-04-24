# Precision Pick-and-Place: Learning Robotic Control through RL

A reinforcement learning project for **dynamic interception under partial observability** with a 7-DOF robotic arm.
The agent learns to track, predict, and grasp a moving object on a conveyor belt, even when the object becomes temporarily unobservable.

## Project Overview

Traditional robotic control assumes full state observability and static environments. In real-world scenarios, however, objects move, system dynamics vary, and critical information may be temporarily unavailable.

This project explores how **reinforcement learning (RL)** can enable a robotic arm to perform interception and grasping when the object is **partially observable**, requiring the agent to rely on **temporal reasoning and prediction** rather than direct perception.

The environment includes:

- a **7-DOF robotic arm**
- a **moving object on a conveyor belt**
- an **occlusion tunnel** that temporarily hides the object
- a **progressive curriculum of increasing difficulty**
- **continuous control** for arm motion and **discrete control** for gripper action

## Motivation

In dynamic manipulation tasks, robots often lose access to complete state information due to occlusions or sensor limitations. Classical control methods struggle in such settings because they rely on continuous, accurate observations.

This project investigates whether RL can learn robust interception policies when:

- the object becomes temporarily unobservable,
- motion must be inferred over time,
- system dynamics such as speed and mass vary,
- explicit motion models are unavailable or unreliable.

The goal is to develop agents capable of **predictive control under uncertainty**, using internal memory to compensate for missing information.

## Problem Statement

Given:

- robot joint angles and velocities,
- gripper state,
- object state **only when visible**,

The agent must:
1. track a moving object,
2. infer its motion during occlusion,
3. predict its future position,
4. intercept it at the correct time,
5. execute a successful grasp.

## Curriculum Design

Training follows a progressive curriculum:

### 1. Baseline
* Constant object velocity
* Fully observable motion
* Simplifies initial learning of interception behavior
 
### 2. Domain Randomization
* Random belt speeds
* Random object mass
* Encourages robustness to dynamic variations###

### 3. Partial Observability
* An occlusion tunnel hides the object temporarily
* Forces the agent to rely on memory and prediction

# Observation Space
The agent observes:

## Proprioceptive State

- 7-DOF joint angles
- joint velocities
- gripper open/close status

## Object State (Partial)

- object position and/or velocity **only when visible**
- no information during occlusion

This creates a **Partially Observable Markov Decision Process (POMDP)**.

# Action Space

The policy outputs:

## Continuous Control

- Cartesian velocity commands:
  - Δx
  - Δy
  - Δz

## Discrete Control

- Gripper action:
  - `0 = Open`
  - `1 = Close`

# Why Reinforcement Learning?

Classical control methods depend on accurate state estimation and predefined motion models. These assumptions break down when observations are missing or unreliable.

RL provides a framework for learning:
- **predictive behavior** from experience,
- **implicit motion models** without explicit equations,
- **memory-based strategies** for handling occlusion.

Recurrent policies such as LSTMs allow the agent to maintain an internal representation of the object’s motion when it is no longer visible.

# Methodology
We train agents using:
- **Proximal Policy Optimization (PPO)**
- **MLP-based policies** for fully observable settings
- **Recurrent policies (PPO + LSTM/GRU)** for partial observability.
Comparisons will be made between:
- feedforward vs recurrent policies,
- performance across curriculum stages.

# Evaluation Metrics 
Performance is evaluated using:
| Metric | Description |
| --- | --- |
| Interception Success Rate (ISR) | Percentage of successful interceptions |
| Grasp Error (Positional) | Distance between gripper center and object at grasp |
| Time-to-Intercept (TTI) | Time taken to reach interception point |
| Path Efficiency Ratio | Ratio of executed trajectory length to optimal trajectory (ideal ≈ 1.0) |
| Velocity Matching Error | Difference between object and end-effector velocity at contact |
 
# Expected Outcomes 
* **Feedforward policies** perform well in fully observable settings but degrade under occlusion.
* **Recurrent policies** learn to:
  - maintain internal state representations,
  - predict object motion during occlusion,
  - improve interception success under uncertainty.
 
# Prerequisites

- Python 3.10+
- PyTorch
- Gymnasium / Isaac Sim / MuJoCo (or equivalent simulator)
- NumPy

# Installation

```bash
git clone https://github.com/sohanpanda104/RL_Vision_based_Interception.git
cd RL_Vision_based_Interception
pip install -r requirements.txt
```

# Future Work

- Longer and more complex occlusion scenarios
- Improved sim-to-real transfer
- Learned state estimation (belief models)
- Multi-object interception
- Comparison with classical state estimation + control pipelines

# Acknowledgements

This project explores reinforcement learning for **dynamic interception under partial observability**, focusing on prediction, memory, and robust control in uncertain environments.
 
