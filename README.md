# Reinforcement Learning for Optimal Taxi Navigation (Taxi-v3)

## Project Overview

This project implements a Reinforcement Learning (RL) agent to solve the classic **Taxi-v3 environment** from the Gymnasium library. The primary goal is to train an autonomous taxi agent to efficiently pick up a passenger from a designated location and drop them off at their desired destination within a simulated 5x5 urban grid. The agent learns to navigate, pick up, and drop off passengers by maximizing cumulative rewards through a trial-and-error process.

The motivation behind this project is to explore fundamental concepts of Reinforcement Learning, specifically Q-learning, and apply them to a practical pathfinding and decision-making problem.

## The Taxi-v3 Environment

The `Taxi-v3` environment is a strategic simulation on a 5x5 grid.

* **Grid:** A 5x5 grid representing an urban area.
* **Taxi:** The agent navigating the grid.
* **Passenger Locations:** Four fixed locations (Red, Green, Yellow, Blue) where a passenger can spawn and needs to be dropped off.
* **Passenger State:** The passenger can be at one of the four locations or inside the taxi.

### Action Space:
The taxi can perform 6 discrete actions:
* `0`: Move South
* `1`: Move North
* `2`: Move East
* `3`: Move West
* `4`: Pickup Passenger
* `5`: Dropoff Passenger

### Observation Space:
Comprises 500 discrete states, encoded as an integer from 0 to 499. Each state uniquely represents the combination of:
* Taxi's position (25 possibilities)
* Passenger's location (4 initial locations + 1 for being in the taxi = 5 possibilities)
* Destination location (4 possibilities)

### Rewards System:
The environment provides feedback to the agent through a reward system:
* `-1`: For each step taken (encourages efficiency).
* `+20`: For a successful passenger dropoff at the correct destination (the primary goal).
* `-10`: For illegal pickup or dropoff actions (discourages invalid moves).
* `-1`: For attempting to move into a wall (incurs time step penalty without position change).

## Reinforcement Learning Approach: Q-Learning

This project uses **Q-learning**, a model-free, off-policy temporal difference reinforcement learning algorithm.

### Key Concepts:

* **Agent:** The taxi.
* **Environment:** The `Taxi-v3` simulation.
* **State ($S_t$):** The current configuration of the environment (taxi position, passenger status, destination).
* **Action ($A_t$):** An operation performed by the taxi.
* **Reward ($R_{t+1}$):** Feedback received from the environment after taking an action.
* **Q-Table:** A lookup table (matrix) storing the estimated maximum cumulative future reward for taking a specific action in a given state, denoted as $Q(s, a)$.
    * Dimensions: `(number_of_states, number_of_actions)` = (500, 6).
    * Initialized with zeros.
* **Bellman Equation (Q-Value Update Rule):** The core formula used to update Q-values iteratively:
    $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$
    * $\alpha$ (Learning Rate): Determines how much new information overrides old Q-values.
    * $\gamma$ (Discount Factor): Balances the importance of immediate vs. future rewards.
* **Exploration-Exploitation Trade-off ($\epsilon$-greedy policy):** A strategy to balance discovering new, potentially better actions (exploration) and taking the best-known action (exploitation).
    * With probability $\epsilon$, a random action is chosen.
    * With probability $1-\epsilon$, the action with the highest Q-value is chosen.
    * $\epsilon$ typically decays over time.

## Implementation Details

The agent was trained over **2,000 episodes**, with a maximum of **100 actions** per training episode (`max_actions`).

### Hyperparameters:
* `num_episodes`: 2000
* `alpha` (Learning Rate): 0.1
* `gamma` (Discount Factor): 1.0 (undiscounted future rewards, suitable for episodic tasks with clear terminal goals)
* `epsilon` (Initial Exploration Rate): 1.0 (starts with full exploration)
* `epsilon_decay`: 0.99 (multiplicative decay per episode)
* `min_epsilon`: 0.01 (minimum exploration rate to ensure continuous, albeit small, exploration)

### Training Process:
1.  **Initialization:** A Q-table of size 500x6 is initialized with zeros.
2.  **Episodic Training:**
    * For each episode:
        * The environment is reset.
        * The taxi takes steps until it terminates (passenger dropped off) or truncates (max actions reached).
        * At each step:
            * An action is chosen using the $\epsilon$-greedy policy.
            * The environment executes the action, returning a new state, reward, and terminal flags.
            * The Q-table is updated using the Bellman equation.
            * Rewards are accumulated.
        * At the end of the episode, the total reward is recorded, and `epsilon` is decayed.
3.  **Policy Extraction:** After training, the optimal policy is derived by selecting the action with the highest Q-value for each state in the Q-table.

## Results

The trained agent's performance was evaluated in a single test episode, starting with a seed of 42 and limited to a maximum of 16 actions.

* **Test Episode Total Reward:** **8**
* **Steps Taken in Test Episode:** **13**

*(Expected result: `episode_total_reward` should be at least 4, indicating efficient learning.)*

## How to Run

To run this project:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/your-repo-name.git](https://github.com/YourGitHubUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    This project primarily relies on `numpy` and `gymnasium`.
    ```bash
    pip install numpy gymnasium imageio ipython
    ```
3.  **Execute the Python script/Jupyter Notebook:**
    The core logic is contained in a single script.
    ```bash
    python your_script_name.py # If you put it in a .py file
    ```
    *(Alternatively, if in a Jupyter Notebook, run all cells sequentially.)*

## Visualization

The project includes code to visualize the agent's performance during the test episode. A GIF `taxi_agent_performance.gif` is generated, showing the taxi's movements, passenger pickup, and dropoff in real-time.

![Taxi Agent Performance](taxi_agent_behavior.gif)
