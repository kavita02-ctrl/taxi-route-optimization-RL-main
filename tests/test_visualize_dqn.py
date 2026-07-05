from types import SimpleNamespace

import numpy as np

from visualize_dqn import draw_megacity


def test_draw_megacity_renders_expected_canvas_size_and_entities():
    state = SimpleNamespace(
        taxi_x=3,
        taxi_y=4,
        passenger_x=10,
        passenger_y=12,
        dest_x=20,
        dest_y=21,
        passenger_in_taxi=False,
        roadblocks=[(1, 1), (2, 2)],
    )

    img = draw_megacity(state)

    assert img.shape == (800, 800, 3)
    assert img.dtype == np.uint8

    # Roadblock should be drawn as a dark cell.
    assert img[8, 8].tolist() == [50, 50, 50]

    # Passenger should appear as a blue circle pixel near the cell center.
    assert img[100, 84].tolist() == [255, 0, 0]

    # Destination should be drawn as a red cell.
    assert img[168, 168].tolist() == [0, 0, 255]
