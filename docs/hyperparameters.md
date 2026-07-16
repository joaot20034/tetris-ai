# Hyperparameters Explained

This document explains every parameter inside `RLConfig`.

---

# Environment

## max_steps_per_episode

Maximum number of moves before the episode is terminated.

Purpose:

- Prevent infinite games
- Keep training consistent

---

# Learning Rate

```python
learning_rate = 1e-4
```

Controls how quickly the neural network updates.

Higher:

- learns faster
- less stable

Lower:

- slower
- more stable

---

# Replay Buffer

```python
buffer_size = 100000
```

Maximum stored experiences.

Larger buffers provide more diverse training.

---

# Batch Size

```python
batch_size = 256
```

Number of experiences sampled every gradient update.

---

# Gamma

```python
gamma = 0.99
```

Discount factor.

Values close to 1 encourage long-term planning.

---

# Tau

```python
tau = 0.05
```

Soft update coefficient for the Target Network.

Smaller values increase stability.

---

# Target Update Interval

Controls how often the Target Network is updated.

---

# Exploration

## Initial Epsilon

```
1.0
```

100% random actions.

---

## Final Epsilon

```
0.05
```

95% exploitation.

---

## Exploration Fraction

Percentage of training used to decay epsilon.

---

# Reward Weights

## reward_lines_cleared

Positive reward for clearing lines.

---

## reward_tetris_multiplier

Extra bonus for clearing four lines simultaneously.

---

## penalty_hole

Punishes creating holes.

---

## penalty_bumpiness

Punishes uneven surfaces.

---

## penalty_height

Discourages building tall stacks.

---

## penalty_game_over

Large negative reward.

---

## reward_survival

Tiny positive reward every frame.

Encourages staying alive.