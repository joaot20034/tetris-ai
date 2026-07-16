# Project Architecture

This document details the structural design of the **Tetris AI** project. Understanding the architecture is essential for navigating the codebase, extending the simulation, or implementing new Reinforcement Learning algorithms.

---

# Table of Contents

- [1. Architectural Philosophy](#1-architectural-philosophy)
- [2. Directory Structure](#2-directory-structure)
- [3. Module Breakdown](#3-module-breakdown)
  - [The `game/` Directory](#the-game-directory-the-simulation)
  - [The `ai/` Directory](#the-ai-directory-the-brain)
  - [The `ui/` Directory](#the-ui-directory-the-observers)
  - [Artifact Directories](#artifact-directories)
  - [Root Execution Scripts](#root-execution-scripts)
- [4. Design Principles Applied](#4-design-principles-applied)

---

# 1. Architectural Philosophy

The project is built around the **Separation of Concerns (SoC)** principle.

In Reinforcement Learning, the **environment (the game)** must remain completely independent from the **agent (the AI)**.

Mixing game logic with machine learning code creates tightly coupled software that is difficult to debug, impossible to benchmark correctly, and extremely hard to extend with new algorithms.

By separating these domains, the project gains several important advantages.

## Headless Execution

The Tetris engine can simulate millions of frames entirely in memory without rendering graphics.

This dramatically increases training speed because rendering becomes the bottleneck long before neural network inference does.

---

## Swappable AI Algorithms

The AI layer communicates with the game only through the environment interface.

Because of this abstraction, the learning algorithm can easily be replaced.

Examples include:

- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN
- Rainbow DQN
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)

The game engine itself remains unchanged.

---

## Maintainability

Every module has a single responsibility.

Examples:

- A bug in the scoring system never affects neural network training.
- A reward shaping modification never changes collision detection.
- Rendering can be replaced without touching gameplay logic.

This keeps the codebase scalable and easy to understand.

---

# 2. Directory Structure

The repository is organized into independent modules.

```text
tetris-ai/
├── ai/
│   ├── agent.py
│   ├── config.py
│   ├── network.py
│   ├── replay_buffer.py
│   ├── rewards.py
│   └── trainer.py
│
├── game/
│   ├── board.py
│   ├── collision.py
│   ├── game.py
│   ├── pieces.py
│   ├── renderer.py
│   └── scoring.py
│
├── ui/
│   ├── charts.py
│   ├── controls.py
│   └── dashboard.py
│
├── logs/
├── models/
│
├── play.py
├── train.py
└── requirements.txt
```

---

# 3. Module Breakdown

## The `game/` Directory (The Simulation)

The `game/` package contains the complete implementation of modern Tetris.

It is intentionally designed to have **zero dependency** on PyTorch, Stable-Baselines3, Gymnasium, or any Reinforcement Learning concepts.

Its only purpose is to simulate the game.

Think of it as the **physics engine** of the project.

---

### `pieces.py`

Responsible for defining every Tetromino.

Responsibilities include:

- Piece matrices
- Rotation states
- Spawn positions
- Modern 7-Bag randomizer
- Piece generation

The 7-Bag algorithm guarantees fair distribution and prevents extremely unlucky streaks.

---

### `board.py`

Represents the 10×20 Tetris board.

Responsibilities include:

- Grid management
- Locking pieces
- Clearing completed rows
- Updating board state

The board is represented using optimized NumPy arrays for maximum performance.

---

### `collision.py`

Contains pure mathematical functions.

Responsibilities:

- Collision detection
- Boundary checking
- Rotation validation
- Piece placement validation

This module contains no rendering or AI logic.

---

### `scoring.py`

Implements the scoring system.

Examples include:

- Single line clear
- Double
- Triple
- Tetris
- Combo scoring
- Back-to-back bonuses (planned)

Keeping scoring isolated makes experimentation significantly easier.

---

### `renderer.py`

The graphical layer of the game.

This is the **only module inside the game package** that imports Pygame.

Responsibilities:

- Draw board
- Draw Tetrominos
- Draw ghost piece
- Draw next queue
- Draw held piece
- Draw score
- Draw animations

Rendering never modifies game state.

It only visualizes it.

---

### `game.py`

The central orchestrator.

Responsibilities:

- Update game state
- Spawn pieces
- Handle user actions
- Call collision logic
- Update board
- Trigger scoring

Most importantly, it exposes a function similar to:

```python
step(action)
```

This mirrors the Gymnasium API and allows the Reinforcement Learning environment to interact with the simulation seamlessly.

---

# The `ai/` Directory (The Brain)

The AI package contains every component related to Reinforcement Learning.

Unlike the game package, it knows nothing about graphics.

Its only goal is learning.

---

### `config.py`

Stores every hyperparameter inside a dataclass.

Examples include:

- Learning rate
- Gamma
- Batch size
- Replay buffer size
- Exploration schedule
- Reward weights

Centralizing configuration makes experiments reproducible.

---

### `agent.py`

Acts as the bridge between the game engine and Stable-Baselines3.

Responsibilities:

- Convert game state into observations
- Map discrete actions
- Reset environments
- Return rewards
- Return termination signals

It implements the Gymnasium environment interface.

---

### `rewards.py`

Contains the reward shaping system.

Calculates:

Positive rewards

- Survival
- Line clears
- Tetrises

Negative rewards

- Holes
- High stacks
- Bumpiness
- Game over

Reward shaping is isolated so it can evolve independently from gameplay.

---

### `network.py`

Defines the neural network architecture.

Current design:

```
Board

↓

Convolutional Layers

↓

Feature Vector

↓

Metadata

↓

Fully Connected Layers

↓

Q Values
```

The architecture combines:

- CNN for spatial board understanding
- MLP for scalar metadata

Examples of metadata:

- Current piece
- Held piece
- Next piece
- Board statistics

---

### `trainer.py`

Coordinates the training process.

Responsibilities:

- Initialize DQN
- Configure callbacks
- Save checkpoints
- Resume training
- Launch TensorBoard logging

This file contains no game logic.

---

### `replay_buffer.py`

**Current Status**

Planned / Stub

Future implementation will include:

- Prioritized Experience Replay
- Importance Sampling
- Custom sampling strategies

The architecture already reserves a dedicated module for these improvements.

---

# The `ui/` Directory (The Observers)

This package visualizes the learning process.

Unlike the game engine, it does not affect simulation.

Its purpose is monitoring.

---

### `dashboard.py`

Displays real-time information such as:

- Episode
- Current reward
- Best reward
- Current score
- Exploration rate
- Learning rate
- Training speed

Implemented through Stable-Baselines3 callbacks.

---

### `charts.py`

Generates live graphs.

Examples include:

- Reward curve
- Loss curve
- Average score
- Episode duration
- Exploration schedule

Charts are rendered with Matplotlib and transferred into the Pygame interface.

---

### `controls.py`

Allows runtime interaction.

Examples:

- Pause training
- Resume training
- Speed up simulation
- Slow down simulation
- Toggle visualization

This allows training without restarting the application.

---

# Artifact Directories

## `logs/`

Automatically generated.

Contains:

- TensorBoard event files
- Training statistics
- Scalar metrics
- Histograms

Ignored by version control.

---

## `models/`

Automatically generated.

Contains:

- Saved checkpoints
- Neural network weights
- Training snapshots

Allows training to resume at any time.

Ignored by version control.

---

# Root Execution Scripts

## `train.py`

Primary entry point.

Responsibilities:

- Load configuration
- Create environment
- Initialize DQN
- Begin training
- Save checkpoints

---

## `play.py`

Evaluation script.

Loads a trained model.

Disables exploration.

Runs the learned policy visually.

Useful for demonstrations and benchmarking.

---

# 4. Design Principles Applied

## Modularity

Every subsystem is independent.

For example:

The reward function can be completely rewritten without touching:

- collision detection
- rendering
- neural network
- scoring

---

## Scalability

The game uses efficient NumPy operations instead of graphical objects.

This enables high simulation throughput, which is essential for Reinforcement Learning where millions of state transitions are required.

---

## Separation of Concerns

Every module has exactly one responsibility.

Examples:

- `renderer.py` only draws graphics.
- `board.py` only manages board state.
- `trainer.py` only manages learning.
- `rewards.py` only computes rewards.

This greatly improves maintainability.

---

## Type Safety

The project extensively uses Python type hints.

Examples include:

- `dict`
- `tuple`
- `Optional`
- `list`
- `np.ndarray`

Static typing reduces runtime bugs and improves IDE support.

---

## Extensibility

The architecture is intentionally future-proof.

Planned improvements include:

- Double DQN
- Dueling DQN
- Rainbow DQN
- PPO
- A2C
- Prioritized Replay
- Curriculum Learning
- Algorithm comparison dashboard

None of these require changes to the Tetris engine.

---

## Performance

Performance is a first-class design goal.

Strategies include:

- NumPy vectorization
- Headless execution
- GPU-accelerated neural networks (PyTorch)
- Replay buffers
- Batched learning
- Lazy rendering
- Modular callbacks

These optimizations allow the project to scale from educational experiments to long-running training sessions.

---

# Summary

The architecture has been designed with the same principles used in modern Reinforcement Learning frameworks.

The project separates simulation, artificial intelligence, visualization, and configuration into clearly defined modules, making the codebase easier to understand, maintain, test, and extend.

This modular design ensures that future research—such as experimenting with different algorithms, reward functions, or neural network architectures—can be performed with minimal changes to the overall system while preserving clean engineering practices.