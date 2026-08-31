from llm_mappo.env.sensor import (
    sensing_radius_grids, detection_prob, false_alarm_prob, sensing_domain,
)


def test_radius_matches_altitude():
    assert sensing_radius_grids(0) == 1
    assert sensing_radius_grids(1) == 5
    assert sensing_radius_grids(2) == 9


def test_detection_prob_decreases_with_altitude():
    p = [detection_prob(a) for a in range(3)]
    assert p == [0.9, 0.8, 0.7]


def test_sensing_domain_is_within_bounds():
    dom = sensing_domain((10, 10, 1), 1, grid_size=20)
    assert all(0 <= x < 20 and 0 <= y < 20 for x, y in dom)


def test_sensing_domain_contains_center():
    dom = sensing_domain((10, 10, 1), 1, grid_size=20)
    assert (10, 10) in dom
