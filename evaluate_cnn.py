import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import megacity_taxi_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = 6
n_observations = 32

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.fc_global = nn.Linear(7, 64)
        self.fc_combined = nn.Linear(32 * 3 * 3 + 64, 128)
        self.head = nn.Linear(128, n_actions)

    def forward(self, x):
        global_x = x[:, :7]
        spatial_x = x[:, 7:].view(-1, 1, 5, 5)

        s = F.relu(self.conv1(spatial_x))
        s = F.relu(self.conv2(s))
        s = s.view(s.size(0), -1)

        g = F.relu(self.fc_global(global_x))

        combined = torch.cat((s, g), dim=1)
        combined = F.relu(self.fc_combined(combined))
        return self.head(combined)

class MegacityWrapper:
    def __init__(self, grid_size=100, roadblocks=50):
        self.env = megacity_taxi_env.MegacityTaxiEnv(grid_size, grid_size, roadblocks)
        self.grid_size = grid_size

    def _extract_state(self, c_state):
        tx, ty = c_state.taxi_x / self.grid_size, c_state.taxi_y / self.grid_size
        px, py = c_state.passenger_x / self.grid_size, c_state.passenger_y / self.grid_size
        dx, dy = c_state.dest_x / self.grid_size, c_state.dest_y / self.grid_size
        in_taxi = 1.0 if c_state.passenger_in_taxi else 0.0

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

        state_array = np.array([tx, ty, px, py, dx, dy, in_taxi] + local_grid, dtype=np.float32)
        return torch.tensor(state_array, device=device).unsqueeze(0)

    def reset(self):
        c_state = self.env.reset()
        return self._extract_state(c_state)

    def step(self, action):
        c_state, reward, done = self.env.step(action)
        return self._extract_state(c_state), reward, done


def load_model(path, device):
    model = DQN(n_observations, n_actions).to(device)
    model_loaded = False
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model_loaded = True
    except Exception as e:
        print(f"Failed to load model {path}: {e}")
    return model, model_loaded

if __name__ == '__main__':
    model_path = 'megacity_dqn_taxi_cnn.pth'
    env = MegacityWrapper()
    policy_net, ok = load_model(model_path, device)
    if not ok:
        print('Model not loaded, exiting.')
        exit(1)

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < 1000:
        with torch.no_grad():
            qvals = policy_net(state).cpu().numpy().ravel()
            action = int(qvals.argmax())
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Evaluation finished. Steps: {steps} | Total Reward: {total_reward}")
