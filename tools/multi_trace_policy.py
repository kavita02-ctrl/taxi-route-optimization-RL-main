import os
import argparse
import csv
import torch
import sys
import os
# Ensure repository root is on sys.path when running from tools/
sys.path.insert(0, os.getcwd())
from visualize_dqn import DQN, MegacityWrapper, n_actions, n_observations, device


def load_model(path):
    model = DQN(n_observations, n_actions).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model, True
    except Exception as e:
        print(f"Failed to load model {path}: {e}")
        return model, False


def trace_one_episode(policy_net, ep_index, max_steps, repeat_threshold, stuck_no_move_threshold, out_dir):
    env = MegacityWrapper(grid_size=100, roadblocks=50)
    state = env.reset()
    total_reward = 0.0
    step = 0
    no_move_count = 0
    pos_counts = {}

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"trace_ep_{ep_index:03d}.csv")

    outcome = {
        'episode': ep_index,
        'outcome': 'unknown',
        'steps': 0,
        'total_reward': 0.0,
        'stuck_reason': '',
    }

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'taxi_x', 'taxi_y', 'pass_x', 'pass_y', 'dest_x', 'dest_y', 'in_taxi', 'action', 'reward'])

        while step < max_steps:
            c_state = env.env.get_state()
            tx, ty = c_state.taxi_x, c_state.taxi_y

            with torch.no_grad():
                action = int(policy_net(state).argmax(1).item())

            next_state, reward, done = env.step(action)
            total_reward += reward

            writer.writerow([step, tx, ty, c_state.passenger_x, c_state.passenger_y, c_state.dest_x, c_state.dest_y, int(c_state.passenger_in_taxi), action, reward])

            # detect no-move
            new_state = env.env.get_state()
            if (tx, ty) == (new_state.taxi_x, new_state.taxi_y):
                no_move_count += 1
            else:
                no_move_count = 0

            key = (tx, ty, int(c_state.passenger_in_taxi))
            pos_counts[key] = pos_counts.get(key, 0) + 1

            if pos_counts[key] > repeat_threshold:
                outcome['outcome'] = 'stuck_repeat'
                outcome['stuck_reason'] = f'repeat>{repeat_threshold}'
                break

            if no_move_count >= stuck_no_move_threshold:
                outcome['outcome'] = 'stuck_no_move'
                outcome['stuck_reason'] = f'no_move>={stuck_no_move_threshold}'
                break

            state = next_state
            step += 1

            if done:
                outcome['outcome'] = 'done'
                break

    outcome['steps'] = step
    outcome['total_reward'] = total_reward

    return outcome, log_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--model', type=str, default='megacity_dqn_taxi_cnn.pth')
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--repeat-threshold', type=int, default=5)
    parser.add_argument('--stuck-no-move', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='trace_logs')
    args = parser.parse_args()

    policy_net, loaded = load_model(args.model)
    if not loaded:
        print('Warning: model not loaded successfully; traces will use an untrained policy')

    summary = []

    for i in range(1, args.episodes + 1):
        print(f"Tracing episode {i}/{args.episodes}...")
        outcome, path = trace_one_episode(policy_net, i, args.max_steps, args.repeat_threshold, args.stuck_no_move, args.out_dir)
        print(f"Episode {i}: {outcome['outcome']} steps={outcome['steps']} reward={outcome['total_reward']}")
        outcome['log_path'] = path
        summary.append(outcome)

    # write summary
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as sf:
        writer = csv.DictWriter(sf, fieldnames=['episode', 'outcome', 'steps', 'total_reward', 'stuck_reason', 'log_path'])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"Tracing completed. Summary written to {summary_path}")
