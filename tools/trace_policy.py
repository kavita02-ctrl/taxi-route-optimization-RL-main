import os
import time
from types import SimpleNamespace
import sys

# Ensure repository root is on sys.path so local imports succeed when running from tools/
sys.path.insert(0, os.getcwd())

from visualize_dqn import DQN, MegacityWrapper, n_actions, n_observations, device
import torch

MODEL_PATH = "megacity_dqn_taxi.pth"
LOG_PATH = "trace_policy.log"


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return True
    return False


def trace_one_episode(max_steps=2000, stuck_no_move_threshold=10, repeat_threshold=5):
    env = MegacityWrapper(grid_size=100, roadblocks=50)
    policy_net = DQN(n_observations, n_actions).to(device)
    if not load_model(policy_net, MODEL_PATH):
        print("Model not found; aborting trace.")
        return

    state = env.reset()
    total_reward = 0
    step = 0
    no_move_count = 0
    pos_counts = {}

    with open(LOG_PATH, "w") as f:
        f.write("step,taxi_x,taxi_y,pass_x,pass_y,dest_x,dest_y,in_taxi,action,reward\n")

        while step < max_steps:
            # Get raw C++ state for logging
            c_state = env.env.get_state()
            tx, ty = c_state.taxi_x, c_state.taxi_y

            # Choose action deterministically
            with torch.no_grad():
                action = policy_net(state).max(1)[1].item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            f.write(f"{step},{tx},{ty},{c_state.passenger_x},{c_state.passenger_y},{c_state.dest_x},{c_state.dest_y},{int(c_state.passenger_in_taxi)},{action},{reward}\n")

            # detect no-move
            if (tx, ty) == (env.env.get_state().taxi_x, env.env.get_state().taxi_y):
                # If action resulted in no move because of roadblock or invalid action, that counts as no move
                no_move_count += 1
            else:
                no_move_count = 0

            key = (tx, ty, int(c_state.passenger_in_taxi))
            pos_counts[key] = pos_counts.get(key, 0) + 1

            if pos_counts[key] > repeat_threshold:
                print(f"Detected repeated position {key} > {repeat_threshold} times at step {step}")
                f.write(f"# STUCK repeated_position,{key},{step}\n")
                break

            if no_move_count >= stuck_no_move_threshold:
                print(f"Detected no-move for {no_move_count} consecutive steps at step {step}")
                f.write(f"# STUCK no_move,{no_move_count},{step}\n")
                break

            state = next_state
            step += 1

            if done:
                print(f"Episode done at step {step} with reward {total_reward}")
                f.write(f"# DONE,{step},{total_reward}\n")
                break

    print(f"Trace written to {LOG_PATH}")


if __name__ == '__main__':
    trace_one_episode()
