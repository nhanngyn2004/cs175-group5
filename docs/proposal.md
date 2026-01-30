---
layout: default
title: Proposal
---

## Summary of Project

The goal of this project is to study how different reinforcement learning design choices affect an agent’s ability to learn navigation in a simplified Minecraft-like environment. The agent (Steve) starts on a small elevated platform surrounded by lava/void and must reach a diamond placed at a goal location. At each time step, the agent receives observations describing its position and nearby blocks and selects from a limited set of movement actions. The task is episodic: an episode ends when the agent reaches the diamond or falls into lava/void. We will investigate how variations in state representation, reward structure, and learning algorithm influence learning speed, stability, and generalization, rather than focusing solely on task success.

## Project Goals

**Minimum goal:**  
Establish a baseline learning setup using tabular Q-learning in a fixed environment and characterize its learning behavior relative to a random policy. We will analyze reward trajectories, success rates, and convergence patterns to understand basic learning dynamics and failure cases.

This goal focuses on setting up a single, controlled learning scenario and understanding its basic behavior. It requires implementing one standard algorithm (tabular Q-learning), using a fixed environment, and comparing it to a trivial baseline like a random policy. The emphasis is on observing learning curves, convergence patterns, and simple failure cases, not on achieving high performance or robustness. This is considered the minimum because it involves only one algorithm, one environment configuration, and limited experimental variation, yet it still produces meaningful insight into whether learning is occurring at all and establishes a foundation for further analysis.

**Realistic goal:**  
Conduct controlled experiments that vary key reinforcement learning design choices, such as reward structure and start/goal configurations, and analyze how these changes affect learning efficiency, stability, and behavior. We will compare results across conditions to identify which design decisions most strongly influence learning outcomes.

This goal expands the project from a single setup to a set of controlled experiments. By systematically varying reward structures, start/goal positions, or other learning-relevant parameters, the team can compare how different design choices influence learning efficiency, stability, and behavior. This stage requires more careful experimental design, multiple runs, and comparative analysis, but it remains feasible within the time constraints of a quarter. Importantly, it produces insight into why learning changes under different conditions, not just whether it works.

**Moonshot goal:**  
Study generalization and representation effects by training agents under limited environment configurations and evaluating performance on unseen layouts. We will compare tabular Q-learning with Deep Q-Learning to analyze how function approximation changes learning behavior, robustness, and transfer to new environments.

This goal goes beyond learning in a fixed or slightly varied environment and instead asks whether learning transfers to new, unseen situations. Studying generalization across layouts and comparing tabular methods to neural function approximation introduces additional complexity, instability, and tuning challenges. These experiments are harder to get working reliably and often require significant debugging, computational time, and careful interpretation of results. While potentially very insightful, they are not guaranteed to succeed within the project timeline, which is why they are appropriately framed as a moonshot rather than an expectation.

## AI/ML Algorithms

We plan to use reinforcement learning methods including tabular Q-learning and, if time permits, Deep Q-Learning with a neural function approximator. These methods allow us to compare discrete state-action learning with function approximation and study their effects on learning behavior and generalization.

## Evaluation Plan

**Quantitative evaluation** will focus on learning behavior rather than single-run performance. Metrics will include success rate, average episode reward, number of steps per episode, and learning speed as measured by convergence of reward curves. We will compare results across different reward structures, environment configurations, and algorithms, using random and simple heuristic policies as baselines. We expect shaped rewards and richer state representations to improve learning efficiency relative to sparse rewards and simpler representations.

**Qualitative evaluation** will involve visual inspection of agent trajectories, analysis of failure cases (e.g., repeated falls into lava or oscillatory behavior), and examination of training curves to identify instability or overfitting. We will also test the agent on simplified or toy configurations to verify correctness and gain intuition about how learning progresses under different design choices.

## AI Tool Usage

AI tools may be used for code debugging, implementation assistance, and documentation support. All learning algorithms, experiments, and evaluations will be designed and conducted by the team. Any use of AI tools will be documented, and results will be evaluated rather than accepted blindly.