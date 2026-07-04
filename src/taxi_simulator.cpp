#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <tuple>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;

class TaxiEnv {
public:
    int taxi_row;
    int taxi_col;
    int passenger_idx;
    int destination_idx;

    const std::vector<std::pair<int, int>> locs = {
        {0, 0}, // 0: Red (R)
        {0, 4}, // 1: Green (G)
        {4, 0}, // 2: Yellow (Y)
        {4, 3}  // 3: Blue (B)
    };

    std::mt19937 rng;

    TaxiEnv(int seed_val = -1) {
        if (seed_val != -1) {
            rng.seed(seed_val);
        } else {
            rng.seed(std::random_device{}());
        }
        // Initialize to some default valid state
        taxi_row = 0;
        taxi_col = 0;
        passenger_idx = 0;
        destination_idx = 1;
    }

    void seed(int seed_val) {
        rng.seed(seed_val);
    }

    int encode(int row, int col, int pass, int dest) const {
        return ((row * 5 + col) * 5 + pass) * 4 + dest;
    }

    std::vector<int> decode(int state) const {
        int dest = state % 4;
        state /= 4;
        int pass = state % 5;
        state /= 5;
        int col = state % 5;
        int row = state / 5;
        return {row, col, pass, dest};
    }

    int get_state() const {
        return encode(taxi_row, taxi_col, passenger_idx, destination_idx);
    }

    void set_state(int state) {
        auto parts = decode(state);
        taxi_row = parts[0];
        taxi_col = parts[1];
        passenger_idx = parts[2];
        destination_idx = parts[3];
    }

    std::tuple<int, py::dict> reset(int seed_val = -1) {
        if (seed_val != -1) {
            rng.seed(seed_val);
        }
        std::uniform_int_distribution<int> dist_pos(0, 4);
        std::uniform_int_distribution<int> dist_loc(0, 3);

        taxi_row = dist_pos(rng);
        taxi_col = dist_pos(rng);
        passenger_idx = dist_loc(rng);
        do {
            destination_idx = dist_loc(rng);
        } while (passenger_idx == destination_idx);

        int state = get_state();
        py::dict info;
        info["prob"] = 1.0;
        info["action_mask"] = get_action_mask(state);
        return {state, info};
    }

    py::array_t<int8_t> get_action_mask(int state) const {
        auto parts = decode(state);
        int r = parts[0];
        int c = parts[1];
        int p = parts[2];
        int d = parts[3];

        auto mask = py::array_t<int8_t>(6);
        auto buf = mask.mutable_data();

        for (int a = 0; a < 6; ++a) {
            int new_r = r;
            int new_c = c;
            int new_p = p;
            bool term = false;
            int reward = -1;

            compute_transition(r, c, p, d, a, new_r, new_c, new_p, reward, term);
            int next_state = encode(new_r, new_c, new_p, d);
            buf[a] = (next_state != state) ? 1 : 0;
        }
        return mask;
    }

    void compute_transition(int r, int c, int p, int d, int action,
                            int &new_r, int &new_c, int &new_p, int &reward, bool &terminated) const {
        new_r = r;
        new_c = c;
        new_p = p;
        reward = -1;
        terminated = false;

        if (action == 0) { // South
            new_r = std::min(r + 1, 4);
        } else if (action == 1) { // North
            new_r = std::max(r - 1, 0);
        } else if (action == 2) { // East
            bool blocked = (c == 1 && (r == 0 || r == 1)) ||
                           (c == 0 && (r == 3 || r == 4)) ||
                           (c == 2 && (r == 3 || r == 4));
            if (!blocked) {
                new_c = std::min(c + 1, 4);
            }
        } else if (action == 3) { // West
            bool blocked = (c == 2 && (r == 0 || r == 1)) ||
                           (c == 1 && (r == 3 || r == 4)) ||
                           (c == 3 && (r == 3 || r == 4));
            if (!blocked) {
                new_c = std::max(c - 1, 0);
            }
        } else if (action == 4) { // Pickup
            if (p < 4 && r == locs[p].first && c == locs[p].second) {
                new_p = 4;
            } else {
                reward = -10;
            }
        } else if (action == 5) { // Dropoff
            if (p == 4 && r == locs[d].first && c == locs[d].second) {
                new_p = d;
                terminated = true;
                reward = 20;
            } else if (p == 4) {
                int loc_idx = -1;
                for (int i = 0; i < 4; ++i) {
                    if (r == locs[i].first && c == locs[i].second) {
                        loc_idx = i;
                        break;
                    }
                }
                if (loc_idx != -1) {
                    new_p = loc_idx;
                } else {
                    reward = -10;
                }
            } else {
                reward = -10;
            }
        }
    }

    std::tuple<int, double, bool, bool, py::dict> step(int action) {
        int new_r = taxi_row;
        int new_c = taxi_col;
        int new_p = passenger_idx;
        int reward = -1;
        bool terminated = false;

        compute_transition(taxi_row, taxi_col, passenger_idx, destination_idx, action,
                           new_r, new_c, new_p, reward, terminated);

        taxi_row = new_r;
        taxi_col = new_c;
        passenger_idx = new_p;

        int next_state = get_state();
        py::dict info;
        info["prob"] = 1.0;
        info["action_mask"] = get_action_mask(next_state);

        bool truncated = false;

        return {next_state, static_cast<double>(reward), terminated, truncated, info};
    }
};

PYBIND11_MODULE(taxi_simulator, m) {
    m.doc() = "High-performance Taxi-v3 C++ simulator exposed to Python";
    py::class_<TaxiEnv>(m, "TaxiEnv")
        .def(py::init<int>(), py::arg("seed") = -1)
        .def("seed", &TaxiEnv::seed, py::arg("seed_val"))
        .def("reset", &TaxiEnv::reset, py::arg("seed") = -1)
        .def("step", &TaxiEnv::step, py::arg("action"))
        .def("encode", &TaxiEnv::encode, py::arg("row"), py::arg("col"), py::arg("pass"), py::arg("dest"))
        .def("decode", &TaxiEnv::decode, py::arg("state"))
        .def_property("s", &TaxiEnv::get_state, &TaxiEnv::set_state)
        .def_readwrite("taxi_row", &TaxiEnv::taxi_row)
        .def_readwrite("taxi_col", &TaxiEnv::taxi_col)
        .def_readwrite("passenger_idx", &TaxiEnv::passenger_idx)
        .def_readwrite("destination_idx", &TaxiEnv::destination_idx);
}
