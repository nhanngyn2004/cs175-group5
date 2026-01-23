---
layout: default
title: Proposal
---

## Summary of Project

The goal of this project is to train a reinforcement learning agent to navigate a simplified Minecraft environment. The agent (Steve) starts on a small elevated platform surrounded by lava/void and must pick up a diamond placed at a goal location. At each time step, the agent receives observations describing its position and nearby blocks and chooses from a limited set of movement actions. The task is episodic: the episode ends when the agent reaches the diamond or falls into lava/void (dies). The agent is rewarded for successfully reaching the goal and penalized for dying or taking excessive steps. The overall objective is to learn a policy that reliably and efficiently navigates to the diamond.

## Project Goals

Minimum goal: Train an agent that can reach the diamond on a fixed map more often than a random baseline.

Realistic goal: Train an agent that succeeds across multiple start and goal positions and avoids lava reliably.

Moonshot goal: Generalize to multiple platform layouts with obstacles and varying goal locations.

## AI/ML Algorithms

We plan to use reinforcement learning, starting with tabular Q-learning and extending to a neural-network-based Q-function (e.g., Deep Q-Learning) if time permits.

## Evaluation Plan

Quantitative evaluation will measure success rate (percentage of episodes where the agent reaches the diamond), average number of steps per successful episode, and average episode reward. Performance will be compared against baseline agents such as a random policy and a simple greedy policy that moves toward the goal when possible. Qualitative evaluation will include observing the agent’s movement in the environment and reviewing training curves to ensure learning is occurring. 

## AI Tool Usage

AI tools may be used for debugging, code assistance, and documentation support. The team will design, train, and evaluate the AI models ourselves.
