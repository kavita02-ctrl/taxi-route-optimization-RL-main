import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.toy_text.taxi import TaxiEnv as GymTaxiEnv
import taxi_simulator
import numpy as np

class CppTaxiEnv(gym.Env):
    """
    A Gym-compatible wrapper around the high-performance C++ Taxi-v3 simulator.
    Delegates step logic to C++ for speed, while maintaining
    Gymnasium compatibility (spaces, seeding, and pygame rendering).
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.cpp_env = taxi_simulator.TaxiEnv()
        self.gym_env = GymTaxiEnv(render_mode=render_mode)
        
        # Mirror observation and action spaces
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space

        # Get maximum steps from Gymnasium spec (usually 200)
        self.max_episode_steps = getattr(self.gym_env.spec, "max_episode_steps", 200)
        if self.max_episode_steps is None:
            self.max_episode_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset step counter
        self.current_step = 0
        
        # Generate initial state using Gymnasium's generator to match exactly
        state, info = self.gym_env.reset(seed=seed, options=options)
        
        # Sync this state to the C++ environment
        self.cpp_env.s = state
        return state, info

    def step(self, action):
        self.current_step += 1
        
        # Step through the fast C++ backend
        next_state, reward, terminated, _, info = self.cpp_env.step(action)
        
        # Determine truncation based on step count
        truncated = self.current_step >= self.max_episode_steps
        
        # Sync state back to Gymnasium for rendering/info
        self.gym_env.unwrapped.s = next_state
        return next_state, reward, terminated, truncated, info

    def render(self):
        # Synchronize states before rendering
        self.gym_env.unwrapped.s = self.cpp_env.s
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def get_state(self):
        locations = [(0, 0), (0, 4), (4, 0), (4, 3)]
        passenger_idx = self.cpp_env.passenger_idx
        destination_idx = self.cpp_env.destination_idx

        if passenger_idx < 4:
            passenger_x, passenger_y = locations[passenger_idx]
        else:
            passenger_x, passenger_y = self.cpp_env.taxi_col, self.cpp_env.taxi_row

        destination_x, destination_y = locations[destination_idx]

        return type(
            "EnvState",
            (),
            {
                "taxi_x": self.cpp_env.taxi_col,
                "taxi_y": self.cpp_env.taxi_row,
                "passenger_x": passenger_x,
                "passenger_y": passenger_y,
                "dest_x": destination_x,
                "dest_y": destination_y,
                "passenger_in_taxi": passenger_idx == 4,
            },
        )()

    @property
    def s(self):
        return self.cpp_env.s

    @s.setter
    def s(self, value):
        self.cpp_env.s = value
        self.gym_env.unwrapped.s = value

    @property
    def np_random(self):
        return self.gym_env.np_random

    @np_random.setter
    def np_random(self, value):
        self.gym_env.np_random = value

from gymnasium.envs.registration import register
try:
    from gymnasium.envs.registration import registry

    if "Taxi-v4" in registry:
        del registry["Taxi-v4"]
except Exception:
    pass

register(
    id="Taxi-v4",
    entry_point="taxi_simulator_wrapper:TaxiEnvWrapper",
    max_episode_steps=200,
    reward_threshold=8,
)


TaxiEnvWrapper = CppTaxiEnv
