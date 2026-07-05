import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import megacity_taxi_env

# Hyperparameters
n_actions = 6
# Observation size: 7 (coords + in_taxi) + 5x5 local grid = 32
n_observations = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def get_cpp_state(self):
        return self.env.get_state()


# Constants for visualization
CELL_SIZE = 8
GRID_SIZE = 100
IMG_SIZE = CELL_SIZE * GRID_SIZE

COLORS = {
    "roadblock": (50, 50, 50),
    "destination": (0, 0, 255),
    "passenger": (255, 0, 0),
    "taxi_empty": (0, 255, 255),
    "taxi_full": (0, 200, 0),
}


def draw_megacity(state_obj, cell_size=CELL_SIZE, grid_size=GRID_SIZE):
    """Renders the 100x100 megacity to an OpenCV image."""
    img_size = grid_size * cell_size
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    for rx, ry in state_obj.roadblocks:
        if 0 <= rx < grid_size and 0 <= ry < grid_size:
            cv2.rectangle(
                img,
                (rx * cell_size, ry * cell_size),
                ((rx + 1) * cell_size, (ry + 1) * cell_size),
                COLORS["roadblock"],
                -1,
            )

    cv2.rectangle(
        img,
        (state_obj.dest_x * cell_size, state_obj.dest_y * cell_size),
        ((state_obj.dest_x + 1) * cell_size, (state_obj.dest_y + 1) * cell_size),
        COLORS["destination"],
        -1,
    )

    if not state_obj.passenger_in_taxi:
        center_x = int(state_obj.passenger_x * cell_size + cell_size / 2)
        center_y = int(state_obj.passenger_y * cell_size + cell_size / 2)
        cv2.circle(img, (center_x, center_y), max(1, cell_size // 2), COLORS["passenger"], -1)

    taxi_color = COLORS["taxi_full"] if state_obj.passenger_in_taxi else COLORS["taxi_empty"]
    cv2.rectangle(
        img,
        (state_obj.taxi_x * cell_size, state_obj.taxi_y * cell_size),
        ((state_obj.taxi_x + 1) * cell_size, (state_obj.taxi_y + 1) * cell_size),
        taxi_color,
        -1,
    )

    return img


draw_grid = draw_megacity


def load_model(model, path):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            return True
        except Exception as e:
            print(f"Failed to load model: {e}; continuing with untrained model.")
            return False
    return False


if __name__ == "__main__":
    try:
        print("Loading Megacity Environment and DQN Model...")

        env = MegacityWrapper(grid_size=GRID_SIZE, roadblocks=50)
        policy_net = DQN(n_observations, n_actions).to(device)
        # Prefer the CNN-trained model if available
        model_path = "megacity_dqn_taxi_cnn.pth"
        if load_model(policy_net, model_path):
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model file {model_path} not found. Running with untrained policy.")

        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        prev_action = None
        OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

        while not done and step_count < 1000:
            cpp_state = env.get_cpp_state()
            frame = draw_megacity(cpp_state)

            cv2.putText(frame, f"Steps: {step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Reward: {total_reward}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            status = "En Route to Pass" if not cpp_state.passenger_in_taxi else "En Route to Dest"
            cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("Megacity DQN Visualization", frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            with torch.no_grad():
                qvals = policy_net(state).cpu().numpy().ravel()
                top_inds = qvals.argsort()[::-1]
                best = int(top_inds[0])

                # Heuristic: prefer actions that reduce Manhattan distance to current target
                c_state = cpp_state
                if not c_state.passenger_in_taxi:
                    tx, ty = c_state.taxi_x, c_state.taxi_y
                    target_x, target_y = c_state.passenger_x, c_state.passenger_y
                else:
                    tx, ty = c_state.taxi_x, c_state.taxi_y
                    target_x, target_y = c_state.dest_x, c_state.dest_y

                cur_dist = abs(tx - target_x) + abs(ty - target_y)

                chosen = None
                # consider top-k actions from policy and pick one that reduces distance
                for cand in top_inds[:4]:
                    cand = int(cand)
                    # pickup/dropoff allowed as-is
                    if cand in (4, 5):
                        chosen = cand
                        break

                    nx, ny = tx, ty
                    if cand == 0 and ty > 0:
                        ny = ty - 1
                    elif cand == 1 and ty < GRID_SIZE - 1:
                        ny = ty + 1
                    elif cand == 2 and tx < GRID_SIZE - 1:
                        nx = tx + 1
                    elif cand == 3 and tx > 0:
                        nx = tx - 1

                    # skip moves into roadblocks
                    if (nx, ny) in set(c_state.roadblocks):
                        continue

                    new_dist = abs(nx - target_x) + abs(ny - target_y)
                    if new_dist < cur_dist:
                        chosen = cand
                        break

                if chosen is None:
                    # fallback: pick first valid action from policy's ranking (avoid off-grid / roadblocks)
                    chosen = None
                    for cand in top_inds:
                        cand = int(cand)
                        if cand in (4, 5):
                            chosen = cand
                            break

                        nx, ny = tx, ty
                        if cand == 0 and ty > 0:
                            ny = ty - 1
                        elif cand == 1 and ty < GRID_SIZE - 1:
                            ny = ty + 1
                        elif cand == 2 and tx < GRID_SIZE - 1:
                            nx = tx + 1
                        elif cand == 3 and tx > 0:
                            nx = tx - 1
                        else:
                            continue

                        if (nx, ny) in set(c_state.roadblocks):
                            continue

                        chosen = cand
                        break

                    if chosen is None:
                        chosen = best

                action = chosen

            prev_action = action

            state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

        print(f"Simulation Ended. Total Steps: {step_count} | Final Reward: {total_reward}")
        cv2.waitKey(1500)
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    finally:
        cv2.destroyAllWindows()
