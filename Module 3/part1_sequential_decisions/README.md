# Part 1: From bandits to gridworlds

Welcome to the first part of Module 3. In Module 2, you fit reinforcement learning (RL) models to choices on a bandit task: one situation, one choice, and a reward that arrived immediately afterwards. Very little of what you actually do in a day looks like that. Making coffee, crossing a road, or picking up a cube with a robot arm all take a *sequence* of actions, the situation you face next depends on the action you just took, and whatever tells you that it went well tends to arrive only at the very end.

This part is about what you have to add to the bandit picture to cope with all that: states, transitions, and a way of assigning credit to actions whose consequences are delayed. It is designed to introduce these concepts through hands-on learning and experimentation.

You will be working with the [`tutorial_3a.ipynb`](./tutorial_3a.ipynb) - the jupyter notebook with all instructions and code. Although you don't need to know and understand it, feel free to look at the helper code present in [`agents.py`](./agents.py), [`environments.py`](./environments.py), [`simulate.py`](./simulate.py), and [`plots.py`](./plots.py).

## Overview

In this tutorial, we are going to explore two key RL concepts: the environment and the agent. We will be using Python and the Farama Gymnasium library, which provides a variety of pre-made environments and a standardized interface, making it easy to develop and compare RL algorithms as well as build your own environments.

We will start with a simple gridworld-like environment, `FrozenLake-v1`, and two agents: a `RandomAgent` and a `QLearningAgent`. By following the standardized approach, we will see how our agents naturally extend to another classic control environment, `CartPole-v1`.

- **Exploring the Environment:** 
  - We will initialize the environment and explore its observation and action spaces.
  - Understanding these is key to successfully training an RL agent.
- **Training a Random Agent:** 
  - We will start with a `RandomAgent`, which chooses actions randomly from the action space. 
  - This will serve as a baseline and help us understand the environment dynamics.
- **Training a Q-Learning Agent:** 
  - Next, we will train a `QLearningAgent`, which uses the Q-Learning algorithm to learn from its experiences and improve its policy over time.

The interactive nature of this tutorial allows you to see the RL process in action and understand the roles of the environment and the agent, as well as the interaction between them.

## Setup Instructions

All the setup instructions including environment setup, dependencies, and how to run the code are provided in the [parent folder's README](../README.md). Please refer to that before you begin with this tutorial.

Enjoy the journey of learning Reinforcement Learning!
