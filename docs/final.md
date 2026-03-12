---
layout: default
title: Final Report
---

## Project Summary

Reinforcement learning is often evaluated purely on final performance. Did the agent win or not? This project takes a different approach. Rather than building the strongest possible agent, our goal is to understand how specific design choices shape the way an agent learns. We use Project Malmo, a platform developed by Microsoft Research that places AI agents inside the Minecraft game engine, as a controlled and repeatable testbed for studying these learning dynamics.

The task we designed is deliberately simple: a reinforcement learning agent is placed at one end of a narrow obsidian platform (5 blocks wide, 10 blocks long) elevated above a lava field, with a single diamond placed at the far end. The agent must navigate to the diamond without falling off the edge. Every episode ends one of three ways. The agent picks up the diamond (success), falls off the platform into the lava (failure), or runs out of time (timeout). This setup may sound straightforward, but it is not trivial. The platform is narrow enough that random movement frequently leads to falling, and the agent receives no guidance about which direction to move. Without a learning algorithm, the agent has no way to distinguish a good move from a bad one. A purely random agent succeeds only about 33% of the time by chance, meaning there is real room for a learning agent to improve, but also real risk of failure if design choices are poorly made.

This is where the challenge lies. Decisions like how rewards are structured, what information the agent can observe, and which learning algorithm is used all fundamentally change what the agent experiences and how it updates its behavior. By isolating these variables and studying their effects, we gain insight into the mechanics of learning itself.

## Approach

### Environment Setup

The environment is implemented as a custom wrapper around the Project Malmo SDK, exposing a simple `reset()` and `step()` interface consistent with the OpenAI Gym convention. On each call to `reset()`, a new Malmo mission is launched and the agent is placed at position (x=0.5, z=0.5, y=51) at the start of the platform. The diamond is placed at (x=0, z=9, y=51) at the far end. The platform itself is a 5×10 grid of obsidian blocks at y=50, surrounded by lava at y=5.

### State / Observation Space

At each step, the agent receives an observation dictionary containing its current grid position: {x, z, y}, where x and z are rounded to the nearest integer and y is a continuous float used for fall detection. The observation is read from Malmo's ObservationFromFullStats handler, which exposes the agent's position and inventory. No visual input (pixels) is used.

### Action Space

The agent has four discrete actions:

| ID | Action | Malmo Command |
|----|--------|---------------|
| 0 | Move North | move 1 |
| 1 | Move South | move -1 |
| 2 | Move West | strafe -1 |
| 3 | Move East | strafe 1 |

Each action is held for 4 ticks (~0.4 seconds) before motion is stopped, giving the agent time to physically move one block in the chosen direction.

### Reward Structure

The reward function is sparse and terminal — rewards are only given when an episode ends:

- **+1.0** — agent picks up the diamond (success_diamond_picked_up)
- **−1.0** — agent falls off the platform (failure_fell_off_platform)
- **0.0** — episode times out (timeout_max_steps_reached)

No intermediate rewards are given for steps taken. This is intentional for the baseline experiments — it gives us a clean signal to compare against shaped reward variants in future experiments.

### Random Baseline Agent

The first agent we evaluate is a random policy. At every step it selects one of the four actions uniformly at random, with no use of observations, memory, or learning:

```python
action = randint(0, 3)
```

This agent implements only `select_action(obs, info)` and has no `observe()` method, meaning the training harness never calls any update step. It is the canonical no-learning baseline — any learning algorithm should eventually surpass it.

### Training Harness

All agents are run through a shared episode loop in `harness/loop.py`. The loop calls `env.reset()` at the start of each episode, then repeatedly calls `agent.select_action()` and `env.step()` until `done=True`. If the agent implements an `observe(obs, action, reward, next_obs, done, info)` method, the harness calls it after each step to allow learning agents to update. Per-episode statistics — total reward, steps, success, termination reason, and seed — are recorded for every episode.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_episodes | 50–200 | Configurable via JSON config |
| max_steps | 200 | Maximum steps per episode before timeout |
| move_ticks | 4 | Ticks per action (~0.4s of movement) |
| platform_y | 51 | Y-level used for fall detection |
| seed | 42 | Fixed for reproducibility |

Configurations are defined in JSON files under `configs/` (e.g. `configs/random_baseline.json`) and passed to the run script, making all experiments fully reproducible.
