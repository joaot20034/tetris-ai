# Training Pipeline

This document explains how the agent learns.

---

# Step 1

Observe the board.

↓

Generate observation.

---

# Step 2

Feed observation into the neural network.

↓

Predict Q-values.

---

# Step 3

Choose action.

Random (exploration)

or

Best action (exploitation)

---

# Step 4

Execute action.

↓

Environment updates.

---

# Step 5

Receive reward.

---

# Step 6

Store experience.

```
State

Action

Reward

Next State
```

inside Replay Buffer.

---

# Step 7

Sample a random batch.

---

# Step 8

Compute Bellman Targets.

---

# Step 9

Calculate Loss

```
Prediction

vs

Target
```

---

# Step 10

Backpropagation

Update neural network weights.

---

# Step 11

Repeat

Millions of iterations.

---

# Monitoring

Training metrics include:

- Episode reward
- Average reward
- Loss
- Epsilon
- Lines cleared
- Survival time

TensorBoard visualizes these metrics in real time.