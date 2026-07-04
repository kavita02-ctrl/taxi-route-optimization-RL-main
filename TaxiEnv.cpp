#include <vector>
#include <tuple>
#include <random>
#include <stdexcept>

// Structure to hold the environment's state
struct EnvState {
    int taxi_x, taxi_y;
    int passenger_x, passenger_y;
    int dest_x, dest_y;
    bool passenger_in_taxi;
};

class HighPerfTaxiEnv {
private:
    int width, height;
    EnvState state;
    std::vector<std::pair<int, int>> valid_locations; // Pickup/Dropoff hotspots
    std::mt19937 rng; // Fast random number generator

public:
    HighPerfTaxiEnv(int w = 5, int h = 5) : width(w), height(h), rng(std::random_device{}()) {
        // Define standard hotspots (R, G, Y, B equivalents)
        valid_locations = {{0, 0}, {0, w-1}, {h-1, 0}, {h-1, w-1}};
        reset();
    }

    // Reset environment state efficiently using bitwise or basic tracking
    EnvState reset() {
        std::uniform_int_distribution<int> dist_grid_x(0, width - 1);
        std::uniform_int_distribution<int> dist_grid_y(0, height - 1);
        std::uniform_int_distribution<int> dist_loc(0, valid_locations.size() - 1);

        state.taxi_x = dist_grid_x(rng);
        state.taxi_y = dist_grid_y(rng);
        
        auto p_loc = valid_locations[dist_loc(rng)];
        state.passenger_x = p_loc.first;
        state.passenger_y = p_loc.second;

        auto d_loc = valid_locations[dist_loc(rng)];
        while (d_loc == p_loc) { // Ensure destination != pickup
            d_loc = valid_locations[dist_loc(rng)];
        }
        state.dest_x = d_loc.first;
        state.dest_y = d_loc.second;
        state.passenger_in_taxi = false;

        return state;
    }

    // High-performance step function
    // Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
    std::tuple<EnvState, int, bool> step(int action) {
        int reward = -1; // Default step penalty
        bool done = false;

        if (action == 0 && state.taxi_y > 0) state.taxi_y--;        // Move South
        else if (action == 1 && state.taxi_y < height - 1) state.taxi_y++; // Move North
        else if (action == 2 && state.taxi_x < width - 1) state.taxi_x++;  // Move East
        else if (action == 3 && state.taxi_x > 0) state.taxi_x--;   // Move West
        
        else if (action == 4) { // Pickup Action
            if (!state.passenger_in_taxi && state.taxi_x == state.passenger_x && state.taxi_y == state.passenger_y) {
                state.passenger_in_taxi = true;
                reward = 10; // Reward for successful pickup
            } else {
                reward = -10; // Illegal pickup penalty
            }
        }
        else if (action == 5) { // Dropoff Action
            if (state.passenger_in_taxi && state.taxi_x == state.dest_x && state.taxi_y == state.dest_y) {
                state.passenger_in_taxi = false;
                reward = 20; // Big reward for completion
                done = true;
            } else {
                reward = -10; // Illegal dropoff penalty
            }
        }

        return std::make_tuple(state, reward, done);
    }

    EnvState getState() const { return state; }
};
