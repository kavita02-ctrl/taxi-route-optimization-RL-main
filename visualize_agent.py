import cv2
import numpy as np
from taxi_simulator_wrapper import TaxiEnvWrapper

# Define the 4 standard pickup/dropoff locations for a 5x5 grid
LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]
COLORS = {
    "grid": (200, 200, 200),
    "taxi_empty": (0, 255, 255),
    "taxi_full": (0, 200, 0),
    "passenger": (255, 0, 0),
    "destination": (0, 0, 255),
}


def draw_grid(state_obj, cell_size=100, grid_size=5):
    """Draws the environment state onto an OpenCV image canvas."""
    img_size = cell_size * grid_size
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    for x in range(0, img_size, cell_size):
        cv2.line(img, (x, 0), (x, img_size), COLORS["grid"], 2)
        cv2.line(img, (0, x), (img_size, x), COLORS["grid"], 2)

    tx, ty = state_obj.taxi_x, state_obj.taxi_y
    px, py = state_obj.passenger_x, state_obj.passenger_y
    dx, dy = state_obj.dest_x, state_obj.dest_y
    in_taxi = state_obj.passenger_in_taxi

    cv2.rectangle(
        img,
        (dx * cell_size, dy * cell_size),
        ((dx + 1) * cell_size, (dy + 1) * cell_size),
        COLORS["destination"],
        3,
    )
    cv2.putText(
        img,
        "D",
        (dx * cell_size + 30, dy * cell_size + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        COLORS["destination"],
        4,
    )

    if not in_taxi:
        cv2.circle(
            img,
            (px * cell_size + 50, py * cell_size + 50),
            25,
            COLORS["passenger"],
            -1,
        )
        cv2.putText(
            img,
            "P",
            (px * cell_size + 35, py * cell_size + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    t_color = COLORS["taxi_full"] if in_taxi else COLORS["taxi_empty"]
    cv2.rectangle(
        img,
        (tx * cell_size + 10, ty * cell_size + 10),
        ((tx + 1) * cell_size - 10, (ty + 1) * cell_size - 10),
        t_color,
        -1,
    )

    return img


def evaluate_and_visualize(q_table):
    env = TaxiEnvWrapper()
    state, _ = env.reset()
    done = False

    print("Starting visual evaluation...")

    while not done:
        state_obj = env.get_state()
        frame = draw_grid(state_obj)
        cv2.imshow("High-Perf Taxi Agent", frame)

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    cv2.waitKey(1500)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from benchmark import run_training_benchmark

    q_table = run_training_benchmark(episodes=2000)
    evaluate_and_visualize(q_table)
