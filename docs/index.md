---
layout: default
title:  Home
---

## About Project Malmo

This project investigates reinforcement learning by training an agent (Steve) to navigate a simplified Minecraft environment. The agent must reach a diamond on an elevated platform while avoiding falling into lava. We study how different design choices, such as reward structures, state representations, and learning algorithms, affect learning behavior.

## Source Code

[GitHub Repository](https://github.com/TaylorTraan/projectmalmo)

## Reports

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)

## Resources

- [Project Malmo](https://github.com/microsoft/malmo) - Minecraft AI research platform
- [MineRL](https://minerl.io/) - Minecraft reinforcement learning competition
- [OpenAI Gym](https://gymnasium.farama.org/) - RL environment toolkit

## Baseline Results

![Random Agent Baseline](assets/baseline_plots.png)

Over 50 episodes, the random policy agent received a total reward of -1.0 in nearly every episode, indicating the agent fell off the platform almost every time. One episode resulted in a reward of 0.0, suggesting the agent survived until the step limit without dying or collecting the diamond. With no learning mechanism, the agent shows no improvement over time, confirming this as a true baseline for comparison.

## Tabular Q-Learning Results

![Tabular Q-Learning](assets/fig1.png)

After 200 episodes of tabular Q-learning with α = 0.5 and sparse rewards, the agent achieved a 32.5% success rate, significantly outperforming the random baseline (~2%). The learning curve shows the agent transitioning from mostly failures early in training to consistent successes after approximately 100 episodes, demonstrating clear learning behavior.

See the [Status Report](status.html) for detailed analysis of learning rate and reward shaping experiments.
