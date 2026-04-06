# Vision-Based Interception

A reinforcement learning project for **vision-based dynamic interception** with a 7-DOF robotic arm.  
The agent learns to visually track, predict, and grasp a moving object on a conveyor belt under progressively harder conditions, including domain randomization and partial observability.

## Project Overview

Traditional robotic control often assumes static objects and perfect state information. In real industrial settings, objects move, physics can vary, and visual input can be partially blocked. This project explores how **visual RL** can solve interception when the robot must act from raw camera observations instead of exact object coordinates.

The environment is designed around:
- a **7-DOF robot arm**
- a **moving target on a conveyor belt**
- **two camera configurations**: eye-to-hand and eye-in-hand
- a **curriculum of increasing difficulty**
- **continuous control** for arm motion and **discrete control** for gripper action

## Motivation

This project studies whether RL can learn robust interception and grasping behavior when:
- the object’s motion is only partially observable,
- object mass and belt speed can vary,
- the target may disappear temporarily behind an occlusion tunnel,
- classical kinematic solvers become unreliable.

The goal is to train policies that learn **predictive “pixels-to-torques” control** directly from visual input and proprioception.

## Problem Statement

Given:
- RGB-D observations from a fixed or wrist-mounted camera,
- robot joint angles and joint velocities,
- gripper state,

the agent must:
1. track a moving object,
2. predict its future position,
3. intercept it at the right time,
4. grasp it accurately.

## Curriculum Design

The project follows a progressive curriculum:

### 1. Baseline
- Constant linear velocity interception
- Simplest setting for initial policy learning

### 2. Domain Randomization
- Random variations in belt speed
- Random variations in object mass
- Improves robustness to changing dynamics

### 3. Partial Observability
- An occlusion tunnel temporarily hides the object
- Tests whether the agent can maintain memory and prediction without continuous visual tracking

## Camera Setups

### Eye-to-Hand
A fixed global camera view:
- gives a broad overview of the scene
- can suffer from arm occlusion

### Eye-in-Hand
A wrist-mounted camera:
- provides a local, close-up view
- can suffer from motion blur and limited field of view

## Observation Space

The agent observes:

### Visual State
- RGB-D image input from the fixed or wrist camera

### Proprioceptive State
- 7-DOF joint angles
- joint velocities
- gripper open/close status

The agent is **not** given the exact 3D coordinates of the block.

## Action Space

The policy outputs:

### Continuous Control
- 3D Cartesian velocity commands:
  - `Delta x`
  - `Delta y`
  - `Delta z`

### Discrete Control
- Gripper action:
  - `0 = Open`
  - `1 = Close`

## Why Reinforcement Learning?

Classical solvers rely on accurate physics and explicit motion models. That works well in ideal conditions, but becomes unreliable when:
- the object is occluded,
- the object velocity changes unexpectedly,
- friction or mass varies,
- the robot must infer motion from raw pixels.

RL, especially with memory-based methods such as **frame stacking** or **LSTMs**, can learn predictive behavior directly from experience.

## Methodology

We plan to train separate visual RL agents for both camera configurations using methods such as:
- **PPO**
- **CNN-based policies**
- recurrent memory when needed for occlusion handling

Training and testing will be done across the three curriculum stages to compare robustness and adaptability.

## Evaluation Metrics

We evaluate performance using:

- **Interception Success Rate (ISR)**  
  Measures how often the robot successfully intercepts the object.

- **Grasp Error (Positional)**  
  Measures how centered the grasp is. Lower is better.

- **Time-to-Intercept (TTI)**  
  Measures reaction speed. Lower is better.

- **Path Efficiency Ratio**  
  Compares the actual trajectory with the optimal one.  
  Ideal is close to `1.0`.

- **Velocity Matching Error**  
  Measures how well the robot matches the object’s velocity at contact.  
  Lower means smoother interception.

## Expected Outcomes

- **Fixed Camera**: expected to perform better in early stages due to stable global observation, but may need memory-based policies like LSTMs to handle occlusions.
- **Wrist Camera**: expected to support more active vision behavior, where the robot moves to improve visibility inside the tunnel.

## Prerequisites

Python 3.10+  
PyTorch  
Gymnasium / Isaac Sim / MuJoCo / or your chosen robotics simulator  
OpenCV  
NumPy

## Installation

git clone https://github.com/sohanpanda104/RL_Vision_based_Interception.git  
cd RL_Vision_based_Interception  
pip install -r requirements.txt

## Future Work

Recurrent policies for long occlusions  
Better sim-to-real transfer  
Improved active vision strategies  
Comparison with classical control baselines  
Multi-object interception scenarios

## Acknowledgements

This project explores vision-based interception in robotic RL, with a focus on dynamic objects, partial observability, and robust control under uncertainty.
