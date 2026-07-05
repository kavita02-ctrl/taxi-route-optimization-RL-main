import math
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import megacity_taxi_env  # Your high-performance C++ backend!

# Hyperparameters for DQN
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500000
TARGET_UPDATE = 10
LR = 1e-4
MEMORY_SIZE = 1000000
EPISODES = 2000

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory to store transitions for batch training
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # A simple Feed Forward Neural Network
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class MegacityWrapper:
    """Wraps the C++ environment to provide normalized Tensors and Local Vision."""
    def __init__(self, grid_size=100, roadblocks=50):
        self.env = megacity_taxi_env.MegacityTaxiEnv(grid_size, grid_size, roadblocks)
        self.grid_size = grid_size
        
    def _extract_state(self, c_state):
        # Normalize coordinates between 0 and 1 for the Neural Network
        tx, ty = c_state.taxi_x / self.grid_size, c_state.taxi_y / self.grid_size
        px, py = c_state.passenger_x / self.grid_size, c_state.passenger_y / self.grid_size
        dx, dy = c_state.dest_x / self.grid_size, c_state.dest_y / self.grid_size
        in_taxi = 1.0 if c_state.passenger_in_taxi else 0.0

        # Local Vision: 5x5 grid centered on taxi (radius=2)
        rb_set = set(c_state.roadblocks)
        radius = 2
        local_grid = []
        for dy_off in range(-radius, radius + 1):
            for dx_off in range(-radius, radius + 1):
                nx = c_state.taxi_x + dx_off
                ny = c_state.taxi_y + dy_off
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) in rb_set:
                    local_grid.append(1.0)
                else:
                    local_grid.append(0.0)

        # Total observation space: 7 (coords+in_taxi) + 25 = 32
        state_array = np.array([tx, ty, px, py, dx, dy, in_taxi] + local_grid, dtype=np.float32)
        return torch.tensor(state_array, device=device).unsqueeze(0)

    def reset(self):
        c_state = self.env.reset()
        return self._extract_state(c_state)

    def step(self, action):
        # Compute a small shaping reward based on reduction in Manhattan distance
        prev = self.env.get_state()
        if not prev.passenger_in_taxi:
            tgt_x, tgt_y = prev.passenger_x, prev.passenger_y
        else:
            tgt_x, tgt_y = prev.dest_x, prev.dest_y

        prev_dist = abs(prev.taxi_x - tgt_x) + abs(prev.taxi_y - tgt_y)

        c_state, reward, done = self.env.step(action)

        # New distance after the action
        if not c_state.passenger_in_taxi:
            new_tgt_x, new_tgt_y = c_state.passenger_x, c_state.passenger_y
        else:
            new_tgt_x, new_tgt_y = c_state.dest_x, c_state.dest_y

        new_dist = abs(c_state.taxi_x - new_tgt_x) + abs(c_state.taxi_y - new_tgt_y)

        # shaping: positive when we move closer, negative when we move away
        shaping = 0.1 * (prev_dist - new_dist)

        shaped_reward = reward + shaping

        return self._extract_state(c_state), shaped_reward, done

env = None
n_actions = 6
n_observations = 32 # 7 coords+in_taxi + 5x5 local grid = 32

# Policy net is what we train, Target net stabilizes learning
policy_net = None
target_net = None
optimizer = None
memory = None

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # Epsilon decay formula
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # Exploit: Pick action with highest predicted Q-value
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Explore: Random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    # Compute the expected Q values (Bellman Equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN on Megacity Taxi")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes to run")
    parser.add_argument("--save", type=str, default="megacity_dqn_taxi_shaped.pth", help="Path to save trained model")
    args = parser.parse_args()

    env = MegacityWrapper()
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    episodes = args.episodes

    print(f"Starting DQN Training on Megacity Environment for {episodes} episodes...")
    start_time = time.time()

    for i_episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(500): # Max steps per episode to prevent infinite loops
            action = select_action(state)
            next_state, reward, done = env.step(action.item())
            total_reward += reward

            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            optimize_model()

            if done:
                break

        # Update the target network periodically
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (i_episode + 1) % 50 == 0 or i_episode == episodes - 1:
            eps = max(EPS_END, EPS_START * math.exp(-1. * steps_done / EPS_DECAY))
            print(f"Episode {i_episode+1}/{episodes} | Last Reward: {total_reward} | Epsilon: {eps:.3f}")

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    # Save the trained model weights to the requested path
    torch.save(policy_net.state_dict(), args.save)
    print(f"Model saved to {args.save}")