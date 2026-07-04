import time
import numpy as np
from taxi_simulator_wrapper import TaxiEnvWrapper  # Your Gym wrapper


def run_training_benchmark(episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    # Initialize your high-performance custom wrapped C++ environment
    env = TaxiEnvWrapper()

    # Initialize Q-Table (500 states, 6 actions)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    total_steps = 0
    start_time = time.time()

    print(f"Starting training loop for {episodes} episodes over C++ environment...")

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellman update equation
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            state = next_state
            total_steps += 1

    end_time = time.time()
    elapsed = end_time - start_time
    steps_per_sec = total_steps / elapsed

    print("\n--- C++ Environment Benchmark Results ---")
    print(f"Total Episodes: {episodes}")
    print(f"Total Steps Processed: {total_steps}")
    print(f"Total Execution Time: {elapsed:.4f} seconds")
    print(f"Throughput: {steps_per_sec:.2f} steps/second\n")

    return q_table


if __name__ == "__main__":
    trained_q = run_training_benchmark()
