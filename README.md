# Navigation

[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Navigation project.

## Project Setup

### Introduction

Train an agent to navigate (and collect bananas) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

Setup the environment as detailed [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started).

### Requirements

 * [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
 * [NumPy](http://www.numpy.org/)

## Training

Follow the instructions in [Navigation.ipynb](Navigation.ipynb) file to train the agent.

## Report

The associated report is detailed on the [Report.md](Report.md) file.

## Evaluation

Sample evaluation of the agent can be checked on [this video](extra/dqt_test01.mp4) file.