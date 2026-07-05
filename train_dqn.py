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
MEMORY_SIZE = 50000
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
        
        # Local Vision: Check for immediate roadblocks
        rb_set = set(c_state.roadblocks)
        obs_s = 1.0 if (c_state.taxi_x, c_state.taxi_y - 1) in rb_set else 0.0
        obs_n = 1.0 if (c_state.taxi_x, c_state.taxi_y + 1) in rb_set else 0.0
        obs_e = 1.0 if (c_state.taxi_x + 1, c_state.taxi_y) in rb_set else 0.0
        obs_w = 1.0 if (c_state.taxi_x - 1, c_state.taxi_y) in rb_set else 0.0
        
        # Total observation space: 11 dimensions
        state_array = np.array([tx, ty, px, py, dx, dy, in_taxi, obs_s, obs_n, obs_e, obs_w], dtype=np.float32)
        return torch.tensor(state_array, device=device).unsqueeze(0)

    def reset(self):
        c_state = self.env.reset()
        return self._extract_state(c_state)

    def step(self, action):
        c_state, reward, done = self.env.step(action)
        return self._extract_state(c_state), reward, done

env = None
n_actions = 6
n_observations = 11 # 7 standard + 4 obstacle sensors

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
    global env, policy_net, target_net, optimizer, memory, steps_done

    env = MegacityWrapper()
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    print("Starting DQN Training on Megacity Environment...")
    start_time = time.time()

    for i_episode in range(EPISODES):
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
        
        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode+1}/{EPISODES} | Last Reward: {total_reward} | Epsilon: {max(EPS_END, EPS_START * math.exp(-1. * steps_done / EPS_DECAY)):.3f}")

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    # Save the trained model weights
    torch.save(policy_net.state_dict(), "megacity_dqn_taxi.pth")
    print("Model saved to megacity_dqn_taxi.pth")