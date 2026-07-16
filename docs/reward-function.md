# Reward Function

The reward function teaches the agent what good Tetris looks like.

---

# Why Reward Shaping?

Game score alone is sparse.

The AI might play hundreds of games before clearing a line.

Reward shaping accelerates learning.

---

# Positive Rewards

| Event | Reward |
|--------|---------|
| Survive | +0.01 |
| Clear Line | +10 |
| Tetris | ×2 Bonus |

---

# Negative Rewards

| Event | Penalty |
|--------|----------|
| Hole | -0.5 |
| Bumpiness | -0.2 |
| Height | -0.1 |
| Game Over | -50 |

---

# Desired Behaviour

The reward function teaches the AI to:

- Keep the board flat
- Avoid holes
- Build for Tetrises
- Stay alive
- Clear efficiently

---

# Visual Example

Good board

```
██████████
██████████
██████████
██████████
```

Bad board

```
███ ██ ███
██ ████ ██
█ █ ███ ██
```

The second board contains holes and uneven surfaces.

The reward function discourages this.

---

# Balancing Rewards

Reward shaping is iterative.

Too much penalty:

- AI becomes overly cautious.

Too much reward:

- AI exploits unintended strategies.

Finding the right balance is part of reinforcement learning research.