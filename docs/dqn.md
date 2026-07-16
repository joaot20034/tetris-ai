# Deep Q-Networks (DQN)

This document explains the Deep Q-Network (DQN) algorithm used by this project.

---

# Table of Contents

- What is DQN?
- Why not use a Q-Table?
- Neural Network Approximation
- Bellman Equation
- Experience Replay
- Target Network
- Training Loop
- Advantages
- Limitations

---

# What is DQN?

Deep Q-Networks (DQN) combine **Q-Learning** with **Deep Learning**.

Instead of storing every possible state inside a gigantic lookup table, a neural network predicts the expected reward (Q-Value) for every possible action.

```
State

↓

Neural Network

↓

Q Values

↓

Best Action
```

---

# Why not use a Q-Table?

Traditional Q-Learning stores values like this:

| State | Left | Right | Rotate |
|--------|------|-------|---------|
| A | 10 | 5 | 3 |
| B | 2 | 7 | 9 |

This works only when there are very few possible states.

Tetris has more board configurations than atoms in the observable universe.

A Q-Table would be impossible.

---

# Neural Network Approximation

Instead of memorizing states, we approximate the Q-function.

Input:

- Board Grid
- Current Piece
- Hold Piece
- Next Pieces

↓

Hidden Layers

↓

Output:

```
Left        2.4

Right       5.6

Rotate      7.1

Drop         9.8
```

The highest value is selected.

---

# Bellman Equation

The Bellman Equation is the mathematical foundation of Q-Learning.

\[
Q(s,a)=r+\gamma\max Q(s',a')
\]

Where:

- **s** = current state
- **a** = action
- **r** = reward
- **γ** = discount factor
- **s'** = next state

The neural network slowly learns to approximate this equation.

---

# Experience Replay

Instead of learning immediately from every frame,

the agent stores experiences.

```
(State,
 Action,
 Reward,
 Next State)
```

inside a Replay Buffer.

Random batches are sampled later.

Benefits:

- Stable learning
- Better convergence
- Reduced correlation

---

# Target Network

DQN uses two neural networks.

```
Online Network

↓

Training

Target Network

↓

Stable Targets
```

The Target Network updates periodically instead of every iteration.

This greatly stabilizes learning.

---

# Training Loop

1. Observe state
2. Predict Q-values
3. Select action
4. Execute action
5. Receive reward
6. Store transition
7. Sample replay buffer
8. Compute Bellman target
9. Compute loss
10. Backpropagate
11. Repeat

Millions of times.

---

# Advantages

- Learns directly from experience
- Works on huge state spaces
- Doesn't require handcrafted strategies
- Can discover unexpected tactics

---

# Limitations

- Data hungry
- Slow to train
- Sensitive to hyperparameters
- Can become unstable without replay buffers and target networks