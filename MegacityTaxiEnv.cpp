#include <vector>
#include <tuple>
#include <random>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Structure to hold the environment's state, now including dynamic roadblocks
struct EnvState {
    int taxi_x, taxi_y;
    int passenger_x, passenger_y;
    int dest_x, dest_y;
    bool passenger_in_taxi;
    std::vector<std::pair<int, int>> roadblocks;
};

class MegacityTaxiEnv {
private:
    int width, height;
    int num_roadblocks;
    EnvState state;
    std::vector<std::pair<int, int>> valid_locations;
    std::set<std::pair<int, int>> roadblock_set;
    std::mt19937 rng;

public:
    MegacityTaxiEnv(int w = 100, int h = 100, int roadblocks = 50)
        : width(w), height(h), num_roadblocks(roadblocks), rng(std::random_device{}()) {
        valid_locations = {
            {0, 0}, {0, w - 1}, {h - 1, 0}, {h - 1, w - 1},
            {w / 2, h / 2}, {w / 4, h / 4}, {3 * w / 4, 3 * h / 4}
        };
        reset();
    }

    EnvState reset() {
        std::uniform_int_distribution<int> dist_grid_x(0, width - 1);
        std::uniform_int_distribution<int> dist_grid_y(0, height - 1);
        std::uniform_int_distribution<int> dist_loc(0, static_cast<int>(valid_locations.size()) - 1);

        state.taxi_x = dist_grid_x(rng);
        state.taxi_y = dist_grid_y(rng);

        auto p_loc = valid_locations[dist_loc(rng)];
        state.passenger_x = p_loc.first;
        state.passenger_y = p_loc.second;

        auto d_loc = valid_locations[dist_loc(rng)];
        while (d_loc == p_loc) {
            d_loc = valid_locations[dist_loc(rng)];
        }
        state.dest_x = d_loc.first;
        state.dest_y = d_loc.second;
        state.passenger_in_taxi = false;

        state.roadblocks.clear();
        roadblock_set.clear();

        while (roadblock_set.size() < static_cast<size_t>(num_roadblocks)) {
            int rx = dist_grid_x(rng);
            int ry = dist_grid_y(rng);

            std::pair<int, int> rb = {rx, ry};

            if (rb != std::make_pair(state.taxi_x, state.taxi_y) &&
                rb != std::make_pair(state.passenger_x, state.passenger_y) &&
                rb != std::make_pair(state.dest_x, state.dest_y)) {
                roadblock_set.insert(rb);
                state.roadblocks.push_back(rb);
            }
        }

        return state;
    }

    std::tuple<EnvState, int, bool> step(int action) {
        int reward = -1;
        bool done = false;

        int next_x = state.taxi_x;
        int next_y = state.taxi_y;

        if (action == 0 && state.taxi_y > 0) next_y--; // South
        else if (action == 1 && state.taxi_y < height - 1) next_y++; // North
        else if (action == 2 && state.taxi_x < width - 1) next_x++; // East
        else if (action == 3 && state.taxi_x > 0) next_x--; // West

        if (action >= 0 && action <= 3) {
            if (roadblock_set.find({next_x, next_y}) != roadblock_set.end()) {
                reward = -5;
            } else {
                state.taxi_x = next_x;
                state.taxi_y = next_y;
            }
        } else if (action == 4) {
            if (!state.passenger_in_taxi && state.taxi_x == state.passenger_x && state.taxi_y == state.passenger_y) {
                state.passenger_in_taxi = true;
                reward = 20;
            } else {
                reward = -10;
            }
        } else if (action == 5) {
            if (state.passenger_in_taxi && state.taxi_x == state.dest_x && state.taxi_y == state.dest_y) {
                state.passenger_in_taxi = false;
                reward = 50;
                done = true;
            } else {
                reward = -10;
            }
        }

        return std::make_tuple(state, reward, done);
    }

    EnvState getState() const { return state; }
};

PYBIND11_MODULE(megacity_taxi_env, m) {
    py::class_<EnvState>(m, "EnvState")
        .def_readwrite("taxi_x", &EnvState::taxi_x)
        .def_readwrite("taxi_y", &EnvState::taxi_y)
        .def_readwrite("passenger_x", &EnvState::passenger_x)
        .def_readwrite("passenger_y", &EnvState::passenger_y)
        .def_readwrite("dest_x", &EnvState::dest_x)
        .def_readwrite("dest_y", &EnvState::dest_y)
        .def_readwrite("passenger_in_taxi", &EnvState::passenger_in_taxi)
        .def_readwrite("roadblocks", &EnvState::roadblocks);

    py::class_<MegacityTaxiEnv>(m, "MegacityTaxiEnv")
        .def(py::init<int, int, int>(), py::arg("w") = 100, py::arg("h") = 100, py::arg("roadblocks") = 50)
        .def("reset", &MegacityTaxiEnv::reset)
        .def("step", &MegacityTaxiEnv::step)
        .def("get_state", &MegacityTaxiEnv::getState);
}
