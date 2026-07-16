# Reinforcement Learning Fundamentals

This document provides a comprehensive introduction to **Reinforcement Learning (RL)** and the core concepts behind the **Tetris AI** project.

It is intended for software engineers who are comfortable with Python but are new to Artificial Intelligence, Machine Learning, or Reinforcement Learning.

Rather than assuming prior knowledge, this guide explains every important concept used throughout the project, from basic terminology to the mathematical ideas behind Deep Q-Networks.

---

# Table of Contents

- [1. The Machine Learning Landscape](#1-the-machine-learning-landscape)
- [2. The Core Reinforcement Learning Loop](#2-the-core-reinforcement-learning-loop)
- [3. Core Reinforcement Learning Concepts](#3-core-reinforcement-learning-concepts)
- [4. Decision Making and Value Functions](#4-decision-making-and-value-functions)
- [5. Advanced Reinforcement Learning Mechanics](#5-advanced-reinforcement-learning-mechanics)
- [6. The Complete Training Pipeline](#6-the-complete-training-pipeline)
- [7. Summary](#7-summary)

---

# 1. The Machine Learning Landscape

To understand Reinforcement Learning, it is helpful to first understand where it fits within the broader field of Artificial Intelligence.

```
Artificial Intelligence
│
├── Machine Learning
│     │
│     ├── Supervised Learning
│     ├── Unsupervised Learning
│     └── Reinforcement Learning
│
└── Deep Learning
      │
      └── Neural Networks
```

Although these fields are closely related, they solve very different problems.

---

## Artificial Intelligence (AI)

Artificial Intelligence is the broad discipline of building systems capable of performing tasks that normally require human intelligence.

Examples include:

- Understanding speech
- Playing games
- Driving vehicles
- Translating languages
- Recognizing images
- Making decisions

Machine Learning is one approach used to build intelligent systems.

---

## Machine Learning (ML)

Machine Learning is a subset of Artificial Intelligence.

Instead of writing explicit rules, we allow computers to discover patterns directly from data.

For example, instead of programming every possible Tetris strategy manually, we allow the computer to learn which strategies work best through experience.

---

## Deep Learning (DL)

Deep Learning is a specialized branch of Machine Learning that uses **Artificial Neural Networks**.

These networks are capable of approximating highly complex mathematical functions.

They excel at problems involving:

- Images
- Audio
- Language
- Games
- Robotics

In this project, the neural network learns to recognize useful board patterns such as:

- Holes
- Flat surfaces
- Dangerous stacks
- Opportunities for Tetrises

---

## Reinforcement Learning (RL)

Reinforcement Learning differs fundamentally from other Machine Learning paradigms.

There is:

- no labeled dataset
- no correct answer
- no teacher

Instead, an intelligent agent learns by interacting with an environment.

It repeatedly performs actions, receives rewards, and gradually discovers which behaviors maximize long-term success.

---

## Comparing Learning Paradigms

### Supervised Learning

The algorithm learns from labeled examples.

Example:

```
Image

↓

Cat
```

The correct answer is already known.

---

### Unsupervised Learning

The algorithm discovers hidden patterns without labels.

Example:

```
Customer Purchases

↓

Customer Groups
```

There is no correct answer.

The algorithm simply discovers structure.

---

### Reinforcement Learning

The algorithm learns entirely through interaction.

```
State

↓

Action

↓

Reward

↓

Learning
```

No human tells the agent which action is correct.

It discovers the optimal strategy through trial and error.

---

# 2. The Core Reinforcement Learning Loop

Every Reinforcement Learning problem follows the same continuous interaction cycle.

```
                 +---------------------------+
                 |                           |
                 |       Environment         |
                 |      (Tetris Engine)      |
                 |                           |
                 +-------------+-------------+
                               ^
                               |
                            Action
                               |
                               |
Observation + Reward           |
                               |
                               v
                 +-------------+-------------+
                 |                           |
                 |          Agent           |
                 |    (Deep Q-Network)      |
                 |                           |
                 +---------------------------+
```

This loop repeats millions of times during training.

Every iteration allows the neural network to become slightly better.

---

# 3. Core Reinforcement Learning Concepts

## Agent

The **Agent** is the learner.

Its responsibility is making decisions.

In this project, the agent is the Deep Q-Network.

Its objective is simple:

> Maximize the total reward over the entire game.

The agent never manipulates the board directly.

Instead, it chooses actions.

---

## Environment

The Environment is everything outside the agent.

For this project, the environment is the complete Tetris simulation.

It contains:

- Gravity
- Collision detection
- Piece spawning
- Line clearing
- Scoring
- Game over detection

The environment simply follows its rules.

It does not help the agent.

---

## Observation (State)

An Observation is the information the agent receives before making a decision.

For Tetris, the observation might include:

- Board grid
- Current Tetromino
- Held piece
- Next piece
- Board statistics

Example:

```
□□□□□□□□□□

□□□□██□□□□

████████□□

Current Piece: T

Next Piece: I

Hold Piece: L
```

The observation completely describes the current situation.

---

## Action

An Action is a decision made by the agent.

Possible actions include:

- Move Left
- Move Right
- Rotate Clockwise
- Rotate Counter-Clockwise
- Soft Drop
- Hard Drop
- Hold Piece

The environment executes the chosen action.

---

## Reward

A Reward is a numerical signal that indicates how good or bad an action was.

Examples:

```
Clear Line

+5
```

```
Create Hole

-2
```

```
Game Over

-300
```

The agent's objective is **not** to maximize immediate reward.

Instead, it maximizes the **sum of future rewards**.

---

## Episode

An Episode is one complete game.

```
Spawn

↓

Play

↓

Game Over
```

Training consists of thousands (or millions) of episodes.

---

# 4. Decision Making and Value Functions

The agent must estimate which actions will produce the greatest future reward.

Several mathematical concepts help achieve this.

---

## Policy

A Policy is the strategy followed by the agent.

Mathematically:

```
State

↓

Policy

↓

Action
```

The policy improves continuously during training.

---

## Value Function

The Value Function estimates:

> "How good is this state?"

It predicts the expected future reward if the agent starts from that state.

---

## Q-Function

The Q-Function is even more specific.

Instead of evaluating states, it evaluates:

```
State

+

Action
```

It answers:

> "If I perform this action now, how much reward will I receive in the future?"

---

## Q-Value

A Q-Value is the actual numerical prediction.

Example:

```
Move Left

12.4

Move Right

18.6

Rotate

7.9

Hard Drop

25.8
```

The highest Q-Value is selected.

---

# 5. Advanced Reinforcement Learning Mechanics

Modern Reinforcement Learning relies on several techniques to stabilize training and improve learning efficiency.

---

## Exploration vs Exploitation

The agent constantly faces a dilemma.

### Exploration

Trying actions it has never tried before.

Useful for discovering better strategies.

Example:

```
Random Move
```

---

### Exploitation

Choosing the action with the highest predicted reward.

Example:

```
Best Known Move
```

---

### Epsilon-Greedy Strategy

Training begins with:

```
ε = 1.0
```

The agent behaves randomly.

As training progresses:

```
ε

↓

0.05
```

The agent increasingly trusts its neural network.

---

## Discount Factor (Gamma)

Future rewards are discounted.

```
γ = 0

Only immediate reward matters.
```

```
γ = 0.99

Long-term planning matters.
```

A high Gamma teaches the agent to prepare for future situations rather than chasing immediate points.

---

## Reward Shaping

Tetris naturally provides sparse rewards.

Without additional guidance:

```
Random Agent

↓

Dies

↓

Learns Nothing
```

Reward shaping introduces intermediate rewards.

Examples include:

Positive:

- Keep the board flat
- Survive longer
- Clear lines

Negative:

- Create holes
- Build tall stacks
- Cause game over

This dramatically accelerates learning.

---

## Replay Buffer

Training directly on consecutive frames causes unstable learning.

Instead:

```
Experience

↓

Replay Buffer

↓

Random Sampling

↓

Training
```

Each experience contains:

```
(State,
 Action,
 Reward,
 Next State,
 Done)
```

Random sampling breaks temporal correlations and improves convergence.

---

## Target Network

If the neural network constantly updates its own learning target, training becomes unstable.

Instead:

```
Main Network

↓

Target Network

↓

Loss Calculation
```

The Target Network updates only occasionally, providing a stable objective.

---

## Neural Network

A traditional Q-Table is impossible for Tetris because the number of board configurations is astronomically large.

Instead, we approximate Q-values using a Deep Neural Network.

Example architecture:

```
Board

↓

Convolutional Layers

↓

Feature Extraction

↓

Fully Connected Layers

↓

Q Values
```

The network learns general concepts rather than memorizing individual board positions.

---

# 6. The Complete Training Pipeline

Every training step follows the same sequence.

```
Observe State

↓

Choose Action

↓

Execute Action

↓

Receive Reward

↓

Observe Next State

↓

Store Experience

↓

Sample Replay Buffer

↓

Calculate Target Q Values

↓

Compute Loss

↓

Gradient Descent

↓

Update Neural Network

↓

Repeat
```

This process repeats millions of times.

Initially, the agent behaves randomly.

Over time, the neural network gradually discovers increasingly effective strategies.

Eventually, the policy transitions from random behavior to deliberate long-term planning.

---

# 7. Summary

Reinforcement Learning enables an agent to learn through interaction rather than supervision.

Unlike traditional Machine Learning, there are no labeled examples or predefined solutions.

Instead, the agent continuously observes the environment, performs actions, receives rewards, and updates its internal policy to maximize long-term success.

Within this project:

- The **Environment** is the Tetris game engine.
- The **Agent** is a Deep Q-Network.
- **Observations** describe the board state.
- **Actions** manipulate the falling Tetromino.
- **Rewards** guide learning toward desirable behaviors.
- **Episodes** represent complete games.
- **Replay Buffers**, **Target Networks**, and **Reward Shaping** provide the stability necessary for effective Deep Reinforcement Learning.

These concepts form the theoretical foundation for every component implemented throughout the Tetris AI project.