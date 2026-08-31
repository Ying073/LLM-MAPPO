from llm_mappo.env.energy import propulsive_power


def test_power_positive():
    assert propulsive_power(10.0) > 0.0


def test_power_hover_finite():
    assert propulsive_power(0.0) > 0.0
