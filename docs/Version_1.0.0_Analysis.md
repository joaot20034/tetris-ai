**Document Type:** Technical Failure Analysis  
**Target Version:** 1.0.0 (Baseline Vanilla DQN)  
**Total Training Timesteps:** ~2.32 Million  

---

## 1. Executive Summary

The initial training run for the baseline Deep Q-Network (Version 1.0.0) successfully executed over 2.3 million timesteps without software crashes. However, from a Reinforcement Learning perspective, the model failed to converge on an optimal or even functional Tetris playing policy. 

Instead of learning to clear lines, the agent experienced policy collapse and optimized for pathological behaviors: intentional game-overs (tower building) and infinite rotation exploits. This document analyzes the telemetry and defines the root causes of this failure.

---

## 2. Training Telemetry and Evaluation

The failure of the policy is immediately visible in the final training telemetry.

![Training Progress](training_report_2326077_steps.png)

### Graph Analysis
*   **Persistent Negative Return:** The moving average (red line) never crosses into positive territory, indicating that the agent never learned to consistently offset penalties with line-clear rewards.
*   **Policy Degradation:** Between timestep 0 and 1,000,000, the moving average actually degrades from roughly -900 to -1500. The agent learned that its initial random actions were bad, but its newly formulated strategy was mathematically worse according to the reward function.
*   **High Volatility:** The raw episode rewards (pink line) show extreme variance, swinging from -500 to -2500. This indicates complete instability in the value function estimation.

### Observed Agent Behaviors
During the evaluation script execution, the agent consistently demonstrated two failure modes:
1.  **Reward Suicide (Tower Building):** The agent stacks pieces strictly vertically, avoiding any horizontal movement, intentionally topping out the board as fast as possible.
2.  **Infinite Spin (Local Optima):** The agent occasionally pins a piece against the floor or a wall and rapidly rotates it left and right indefinitely without ever locking it.

---

## 3. Root Cause Analysis

The observed behaviors are not random bugs; they are highly optimized responses to mathematical flaws in the environment design and reward shaping.

### Root Cause A: The Absolute Penalty Trap (Reward Shaping)
The agent was penalized on every single step based on the total number of holes, bumpiness, and height of the board. 

If a board contains 5 holes, the agent receives a penalty of -5 on step $t$, another -5 on step $t+1$, and another -5 on step $t+2$. The mathematical consequence is that merely surviving longer generates an exponentially larger negative score. The agent correctly deduced that the mathematical optimal policy is to end the game immediately to stop the accumulation of step-based penalties.

*   **Required Fix:** Shift from **Absolute Rewards** to **Delta Rewards**. The agent must only be penalized for the change in state ($\Delta$).
    $$R_{penalty} = State_{t-1} - State_t$$
    If the agent inherits a board with 5 holes and makes a move that results in 5 holes, the penalty should be 0, not -5.

### Root Cause B: Infinite Lock Delay (Environment Mechanics)
The agent discovered that locking a piece inevitably increases board height (and potentially bumpiness/holes), which triggers the flawed reward penalties mentioned above. 

To avoid locking, the agent exploits the rotation mechanics. If rotating a piece resets the gravity timer or prevents the piece from locking to the floor, the agent can spin the piece infinitely. This prevents the state from advancing and temporarily pauses all negative reward accumulation.

*   **Required Fix:** Implement a strict lock-delay limit. If a piece's $y$-coordinate does not change over a set number of frames, it must forcefully lock regardless of rotational inputs.

### Root Cause C: Vanilla DQN Overestimation
Standard DQN algorithms are known to suffer from maximization bias, where they overestimate the Q-values of suboptimal actions. In a stochastic environment like Tetris, this causes the neural network to confidently commit to poor topological placements. 

Furthermore, the exploration fraction was set to 0.2, meaning the agent stopped exploring randomly at approximately 400,000 steps. It spent the remaining 1.9 million steps exploiting a severely undertrained, overestimated Q-table.

*   **Required Fix:** Upgrade the algorithmic architecture to Double DQN to separate action selection from action evaluation, and extend the exploration fraction to at least 0.5.

---

## 4. Conclusion

Version 1.0.0 successfully validated the software architecture (Gymnasium integration, neural network forwarding, Pygame rendering, and TensorBoard logging). However, the mathematical definition of the environment failed. The agent successfully optimized the environment, but the environment was incentivizing the wrong behaviors.