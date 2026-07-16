# Tetris AI

> A Reinforcement Learning framework where an AI agent learns to master modern Tetris from scratch using Deep Q-Networks (DQN).

**Author:** [João Consciência Trindade]  
**License:** [MIT License]  
**Version:** [v1.0.0]  
**Last Updated:** [July 16, 2026]  

---

## Table of Contents

The documentation for this project is divided into comprehensive modules. This document serves as the project overview.

*   [1. Introduction and Overview](README.md)
*   [2. Project Architecture](architecture.md)
*   [3. Reinforcement Learning Fundamentals](reinforcement-learning.md)
*   [4. Why Tetris?](why-tetris.md)
*   [5. Deep Q-Networks (DQN)](dqn.md)
*   [6. Hyperparameters](hyperparameters.md)
*   [7. Reward Function](reward-function.md)
*   [8. Training Pipeline](training.md)
*   [9. Roadmap and Planned Features](roadmap.md)
*   [10. Glossary and FAQ](glossary.md)
*   [11. References](references.md)

---

## 1. Introduction

### What is this project?
Tetris AI is a complete, custom-built framework designed to train a reinforcement learning agent to play modern Tetris. The project encompasses a pure Python implementation of the Tetris game engine, a standardized Gymnasium environment, and a deep learning training pipeline utilizing Deep Q-Networks (DQN). 

Crucially, this is an autonomous reinforcement learning project. The agent operates entirely without scripted strategies, heuristics, or hardcoded moves. It begins with zero knowledge of the game's mechanics and learns exclusively through trial, error, and interaction with the environment's reward system.

### Why was it created?
This framework was created to serve as a rigorous testbed for reinforcement learning architectures. Tetris presents a notoriously difficult environment for artificial intelligence due to its requirement for long-term planning, its massive state space, and the delayed nature of its rewards. Building this project bridges the gap between theoretical deep learning concepts and practical, software engineering execution.

### Who is it for?
This handbook and the accompanying source code are designed for software engineers, university students, and machine learning practitioners who possess a strong foundation in Python but may be encountering applied Reinforcement Learning for the first time. It is structured to be both a functional research environment and a comprehensive educational resource.

### Current Development Status
The end-to-end training pipeline is fully operational. The current implementation includes a headless game engine, a modular architecture, a custom feature extractor combining Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP), and a real-time visualization dashboard. The DQN agent is currently capable of learning the mechanics of the game and actively clearing lines.

---

## 2. Project Goals

The project is driven by both educational and strict software engineering objectives.

### Engineering Goals
*   **Build a complete modern Tetris engine:** Develop a fast, dependency-free Python simulation featuring the 7-bag randomizer, ghost pieces, and standard scoring.
*   **Train an autonomous RL agent:** Successfully implement a deep learning agent capable of sustained survival and line clearing without human intervention.
*   **Produce reusable research architecture:** Construct the codebase using the Model-View-Controller (MVC) pattern, allowing headless training at millions of frames per second while supporting visual rendering on demand.
*   **Support future experimentation:** Design the agent interfaces to strictly adhere to Gymnasium standards, allowing seamless swapping of algorithms (e.g., transitioning from DQN to Proximal Policy Optimization).

### Educational Goals
*   **Demonstrate state representation:** Illustrate how a complex 2D grid and 1D queue can be translated into mathematical tensors for neural network consumption.
*   **Visualize learning in real-time:** Provide a transparent look into the agent's decision-making process via integrated Matplotlib charting and live Pygame rendering.
*   **Compare algorithms [Planned]:** Establish baseline metrics using DQN to eventually compare against more advanced algorithms like Rainbow DQN or PPO.

---

## 3. Technologies

The technology stack was selected to prioritize performance during training, modularity, and industry-standard machine learning practices. 

| Library | Purpose |
| :--- | :--- |
| **Python** | The core programming language used for the simulation and training scripts. |
| **PyTorch** | The deep learning framework used to build and update the neural networks (CNN and MLP). |
| **Gymnasium** | The standard API used to bridge the custom Tetris game engine with reinforcement learning algorithms. |
| **Stable-Baselines3** | A reliable implementation of reinforcement learning algorithms (including DQN), providing the core learning loops and replay buffers. |
| **NumPy** | Used for high-performance matrix mathematics, specifically handling the 2D grid logic and tensor preprocessing. |
| **Pygame** | The rendering engine used to visualize the agent's gameplay and overlay the training dashboard. |
| **TensorBoard** | The logging and visualization suite used to track loss metrics, average rewards, and episode lengths over millions of steps. |
| **Matplotlib** | Utilized alongside Pygame to render real-time reward curve charts directly onto the training dashboard. |

---

## 4. Conclusion

The Tetris AI project establishes a robust, scalable foundation for reinforcement learning research. By enforcing strict separation of concerns between the game logic, the neural network models, and the visualization tools, the framework succeeds in transforming a classic video game into a mathematically rigorous deep learning problem. 

The subsequent documentation files in this handbook will break down exactly how this architecture is constructed, the mathematics driving the deep learning models, and how to operate the training pipeline.