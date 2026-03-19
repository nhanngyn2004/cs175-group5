---
layout: default
title: Final Report
---

## Project Summary

Reinforcement learning is often evaluated purely on final performance. Did the agent win or not? This project takes a different approach. Rather than building the strongest possible agent, our goal is to understand how specific design choices shape the way an agent learns. We use Project Malmo, a platform developed by Microsoft Research that places AI agents inside the Minecraft game engine, as a controlled and repeatable testbed for studying these learning dynamics.

The task we designed is deliberately simple: a reinforcement learning agent is placed at one end of a narrow obsidian platform (5 blocks wide, 10 blocks long) elevated above a lava field, with a single diamond placed at the far end. The agent must navigate to the diamond without falling off the edge.

![Environment Setup](assets/image.png) 

Every episode ends one of three ways. The agent picks up the diamond (success), falls off the platform into the lava (failure), or runs out of time (timeout). This setup may sound straightforward, but it is not trivial. The platform is narrow enough that random movement frequently leads to falling, and the agent receives no guidance about which direction to move. Without a learning algorithm, the agent has no way to distinguish a good move from a bad one. A purely random agent succeeds only about 33% of the time by chance, meaning there is real room for a learning agent to improve, but also real risk of failure if design choices are poorly made.

This is where the challenge lies. Decisions like how rewards are structured, what information the agent can observe, and which learning algorithm is used all fundamentally change what the agent experiences and how it updates its behavior.

A hand-coded solution might seem possible—just move forward until you reach the diamond. But the agent has no built-in knowledge of the environment's layout, where the edges are, or which direction leads to the goal. It receives only its current (x, z) coordinates at each step. Mapping these coordinates to optimal actions requires either human-engineered rules (which defeats the purpose of studying learning) or a learning algorithm that discovers the state-action relationship through experience. Reinforcement learning provides exactly this: a framework for learning optimal behavior through trial-and-error interaction with an unknown environment.

Our central question is not simply whether the agent can learn to succeed, but how different design choices shape that learning. We investigate how sparse rewards compare to shaped rewards with step penalties, how learning rate affects convergence speed and stability, and how much better a learning agent can perform relative to a random baseline. By isolating these variables and studying their effects, we gain insight into the mechanics of reinforcement learning itself—insight that generalizes beyond this specific task to inform how practitioners should approach RL design decisions in other domains.

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

No intermediate rewards are given for steps taken. This is intentional for the baseline experiments, as it gives us a clean signal to compare against shaped reward variants in future experiments.

### Random Baseline Agent

The first agent we evaluate is a random policy. At every step it selects one of the four actions uniformly at random, with no use of observations, memory, or learning:

```python
action = randint(0, 3)
```

This agent implements only `select_action(obs, info)` and has no `observe()` method, meaning the training harness never calls any update step. It is the canonical no-learning baseline, where any learning algorithm should eventually surpass it.

### Training Harness

All agents are run through a shared episode loop in `harness/loop.py`. The loop calls `env.reset()` at the start of each episode, then repeatedly calls `agent.select_action()` and `env.step()` until `done=True`. If the agent implements an `observe(obs, action, reward, next_obs, done, info)` method, the harness calls it after each step to allow learning agents to update. Per-episode statistics, total reward, steps, success, termination reason, and seed, are recorded for every episode.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_episodes | 50–200 | Configurable via JSON config |
| max_steps | 200 | Maximum steps per episode before timeout |
| move_ticks | 4 | Ticks per action (~0.4s of movement) |
| platform_y | 51 | Y-level used for fall detection |
| seed | 42 | Fixed for reproducibility |

Configurations are defined in JSON files under `configs/` (e.g. `configs/random_baseline.json`) and passed to the run script, making all experiments fully reproducible.

### Tabular Q-Learning Algorithm

We implemented standard tabular Q-learning, a model-free reinforcement learning algorithm that learns an action-value function Q(s,a) representing the expected cumulative reward from taking action a in state s and following the optimal policy thereafter (Sutton & Barto, 2018). The Q-table is updated after each transition using the Bellman equation:

**Q(s,a) ← Q(s,a) + α [ r + γ max_a′ Q(s′,a′) − Q(s,a) ]**

where:
- **α (learning rate)**: Controls how much new information overwrites the existing Q-value estimate
- **γ (discount factor)**: Set to 0.99, determines the importance of future rewards relative to immediate rewards
- **r**: The immediate reward received after taking action a in state s
- **max_a′ Q(s′,a′)**: The maximum Q-value achievable from the next state s′

The Q-table is initialized to zeros for all state-action pairs. States are discretized as (x, z) grid positions, yielding a manageable state space of approximately 50 unique states (5 × 10 platform).

### Exploration Strategy

Balancing exploration and exploitation is critical in reinforcement learning. We use an **epsilon-greedy policy** with linear decay:

- **Initial ε = 0.5**: The agent explores randomly 50% of the time at the start of training
- **Final ε = 0.05**: Exploration decreases to 5% by the end of training
- **Decay schedule**: Linear decay over 150 episodes

At each step, the agent selects a random action with probability ε, or the action with the highest Q-value with probability (1 − ε). This strategy ensures sufficient exploration early in training while shifting toward exploitation as the Q-values converge.

### Hyperparameter Experiments

To study how design choices affect learning, we systematically varied two key factors:

**Learning Rate Comparison (α)**

Smith (2017) establishes α = 0.1 as a standard baseline for iterative optimization, noting that learning rate is a highly sensitive hyperparameter. We compare:

| Learning Rate | Update Magnitude | Expected Behavior |
|--------------|------------------|-------------------|
| α = 0.1 | 10% new, 90% retained | Slower, more stable convergence |
| α = 0.5 | 50% new, 50% retained | Faster adaptation, higher variance |

**Reward Shaping Comparison**

| Reward Scheme | Diamond | Fall | Step | Design Intent |
|---------------|---------|------|------|---------------|
| Sparse | +1.0 | −1.0 | 0.0 | Clean signal, delayed feedback |
| Step Penalty | +1.0 | −1.0 | −0.01 | Encourage efficiency |

The step penalty was intended to discourage slow, meandering behavior. However, as documented in the Evaluation section, this shaping inadvertently incentivized risky, short-duration episodes.



## Evaluation 

### Evaluation Setup

We evaluate our agent under different reward structures and learning rates to understand how these factors influence performance and behavior. Each experiment is run for 200 episodes using a fixed random seed. We compare a baseline agent with no learning, as well as Q-learning agents trained under sparse reward and step-penalty reward configurations with learning rates α = 0.1 and α = 0.5.

Performance is measured using total reward per episode and episode length, allowing us to capture both task success and behavioral differences across configurations. These metrics provide both a quantitative view of performance and a qualitative understanding of how the agent behaves under different training conditions.

---

### Baseline Performance

We first examine the behavior of a baseline agent with no learning.

![Baseline Performance](assets/baseline_plots.png)

The baseline agent performs poorly, with highly inconsistent rewards and frequent failures. This establishes a reference point, showing that without learning, the agent is unable to consistently reach the goal and performs close to random behavior.

This comparison is important because it confirms that any improvements observed in later experiments are due to learning rather than chance. It also highlights the difficulty of the environment, as even random exploration rarely leads to successful outcomes.

---

### Learning Performance Across Configurations

Smith (2017) uses an initial learning rate of 0.1 as a baseline in evaluating training performance across multiple architectures, treating it as a standard reference point for comparison against alternative learning rate strategies. The study demonstrates that learning rate is a highly sensitive hyperparameter and that performance can vary significantly as it changes. Because 0.1 is used in the literature as a conventional baseline value for iterative optimization, we adopt α = 0.1 as a stable and principled reference point in our experiments before evaluating more aggressive alternatives such as α = 0.5. 

Source: Smith, Leslie N. "Cyclical learning rates for training neural networks." 2017 IEEE winter conference on applications of computer vision (WACV). IEEE, 2017.

We next evaluate how learning improves performance under different reward structures and learning rates.

#### Sparse Reward

<p align="center">
  <img src="assets/spare_lr0.1.png" width="83%" style="margin-left: 9%;">
</p>

<p align="center">
  <img src="assets/sparse_lr0.5.png" width="70%">
</p>

Under the sparse reward configuration, both learning rates show improvement over time, as the agent gradually learns to reach the diamond. The α = 0.1 setting produces smoother and more stable learning, while α = 0.5 exhibits greater variability due to larger update steps.

These trends suggest that while both configurations are capable of learning, stability plays a critical role in long-term performance. The smoother curves observed with α = 0.1 indicate more consistent policy improvement, whereas α = 0.5 reacts more strongly to individual experiences, leading to fluctuations in performance.

---

#### Step Penalty Reward

<p align="center">
  <img src="assets/step_lr0.1.png" width="83%" style="margin-left: 9%;">
</p>

<p align="center">
  <img src="assets/step_lr0.5.png" width="70%">
</p>

Under the step-penalty configuration, performance is generally lower and more inconsistent. While the agent receives more frequent feedback, the negative reward at each step encourages shorter episodes, often leading to premature failure rather than successful completion of the task.

This indicates that more frequent feedback does not necessarily improve learning outcomes. Instead, the added penalty shifts the objective toward minimizing time steps rather than maximizing success, which can lead to unintended and suboptimal behaviors.

---

### Episode Length Analysis

To better understand behavioral differences, we compare episode lengths across reward structures.

![Episode Length Comparison](assets/episode_lengths.png)

Episodes under the sparse reward configuration tend to be longer, reflecting more exploration before success or failure. In contrast, episodes under the step-penalty configuration are significantly shorter, indicating that the agent is incentivized to terminate episodes quickly.

Importantly, shorter episodes do not necessarily indicate better performance. In this case, they often correspond to the agent falling off the platform early rather than efficiently reaching the goal. This difference highlights how the reward function directly influences the agent’s strategy, leading to more aggressive but less reliable behavior under step penalties.

---

### Learning Rate Effects

Across both reward settings, the learning rate plays a key role in stability. The lower learning rate (α = 0.1) results in more consistent and stable performance, while α = 0.5 leads to faster but more volatile learning.

This tradeoff reflects a fundamental aspect of reinforcement learning, where larger updates can accelerate learning but reduce stability, especially in environments with stochastic transitions. As a result, α = 0.1 provides more reliable improvement over time, even if it learns more slowly in the early stages.

---

### Main Trade Offs

<img src="assets/table_comparison.png" width="70%"/>

The results reveal a clear tradeoff between accuracy and efficiency across the two reward configurations. The sparse reward model achieved a higher success rate (32.5%), indicating that it more reliably learned how to reach the objective. This suggests that, despite slower learning and longer episodes, the agent was able to develop a more effective and goal-oriented policy.

In contrast, the step penalty model exhibited lower accuracy, with a success rate of only 16.5%, but demonstrated greater efficiency in terms of episode length (9.7 steps on average compared to 16.4 under sparse rewards). However, this apparent efficiency is misleading. The shorter episodes are largely due to premature termination from falling, rather than the agent successfully navigating to the goal. As a result, the step penalty configuration encourages faster but less reliable behavior.

Overall, these findings highlight that while reward shaping can improve efficiency in terms of shorter episodes, it may come at the cost of reduced accuracy and suboptimal policy learning. In this task, the sparse reward structure ultimately leads to more successful outcomes, even if it requires more time for the agent to learn and execute its strategy.

### Failure Modes

Despite improvements in learning, the agent exhibits several consistent failure modes. In many episodes, the agent falls off the platform near the goal, suggesting that it has learned a risky but direct policy. Under the step-penalty configuration, the agent frequently terminates episodes early, prioritizing shorter trajectories over successful completion.

These failure cases demonstrate that the agent’s learned policy is sensitive to small changes in state and may not generalize well across different situations. This indicates that the agent often learns locally optimal strategies that do not fully align with the desired objective.

---

### Limitations

Our evaluation is limited by the use of a single random seed and a relatively small number of training episodes. Reinforcement learning is inherently stochastic, and results may vary across runs.

Additionally, the tabular Q-learning approach used in this project does not scale well to larger or more complex environments. These limitations suggest that further experimentation, including multiple seeds and more advanced methods, would be needed to draw stronger conclusions.

---

### Overall Insights

Taken together, our results show that reward design has a greater impact on agent behavior than learning rate alone. While tuning the learning rate affects stability, the choice of reward function fundamentally shapes the strategy learned by the agent.

The sparse reward setting, although more challenging, ultimately leads to more reliable success once the agent discovers an effective strategy. In contrast, the step-penalty setting encourages faster but riskier behavior, often resulting in premature failure.

Overall, the agent is able to learn a policy that improves performance over the baseline, but does not achieve fully stable or optimal behavior within the given training horizon. These findings highlight the importance of carefully designing reward functions in reinforcement learning tasks.



## Resources Used

This project relies on a small set of tools and references to implement and evaluate reinforcement-learning behavior in a Minecraft-based environment.

### Libraries and Frameworks

- **Project Malmo (`MalmoPython`, Microsoft Research)**: The primary RL environment and API used to run missions in Minecraft and obtain observations and rewards. It enabled us to study agent behavior in a controlled task setting.
- **matplotlib (optional)**: Used to produce simple visual summaries of results (e.g., reward trends across episodes) for reporting.

### Platforms and Environment

- **Docker + `andkram/malmo` image**: Provided a consistent runtime with Minecraft, Malmo, and Python preconfigured. This reduced setup overhead and improved reproducibility across machines.
- **Minecraft (Malmo mod)**: Served as the simulator where the agent navigates a platform to reach the goal (diamond pickup). It provided the interactive dynamics required for the task.
- **noVNC / Jupyter Notebook**: Used for visualization and interactive testing (monitoring Minecraft, running small exploratory checks) during development.

### Algorithms and Conceptual Resources

- **Tabular Q-learning**: Implemented using the standard Bellman update with epsilon-greedy exploration to learn an action-value table over discrete states.
- **Baseline policies (random and fixed-direction)**: Used as non-learning reference policies to contextualize learning performance.
- **Gym-style environment design (conceptual)**: Used as an organizing interface (`reset`/`step`) for consistent experiments; the project does not depend on the `gym` package.

### Key Python Utilities

- **`json`**: Used for reading experiment configs and handling structured inputs/outputs.
- **`random`**: Used for baseline action selection and exploration in Q-learning.
- **`argparse`**: Used to provide clear command-line interfaces for running experiments.

### AI Tools

- **ChatGPT and Cursor (AI assistants)**: Used to clarify RL/Malmo concepts, refine report-style wording, and suggest boilerplate patterns (e.g., CLI/logging structure). Their influence appears primarily in documentation phrasing, script structure, and minor helper code; the final algorithms, environment setup, and results were implemented and verified by the project team.

### Copied/Adapted Code

- **External code reuse**: No external code was directly copied; implementations follow standard reinforcement-learning and Project Malmo usage patterns.
