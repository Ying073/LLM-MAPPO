from llm_mappo.dpes.pheromone import classify_grid, update_pheromone
from llm_mappo.utils.config import DPESConfig


def test_classify_high_value():
    assert classify_grid(p=0.5, last_visit=0, t=10, cfg=DPESConfig()) == "hv"


def test_classify_long_unvisited():
    cfg = DPESConfig()
    assert classify_grid(p=0.0, last_visit=0, t=cfg.revisit_threshold + 1, cfg=cfg) == "lu"


def test_classify_confirmed():
    cfg = DPESConfig()
    assert classify_grid(p=0.95, last_visit=9, t=10, cfg=cfg) == "cs"


def test_update_pheromone_resets_lu_on_visit():
    cfg = DPESConfig()
    # 长期未访网格：访问后信号清零
    dp = update_pheromone("lu", dp=0.003, released=0.0, diffused=0.0, visited=True, cfg=cfg)
    assert dp == 0.0


def test_update_hv_evaporates_and_diffuses():
    cfg = DPESConfig()
    dp = update_pheromone("hv", dp=0.5, released=cfg.release_hv, diffused=0.02, visited=False, cfg=cfg)
    assert dp == cfg.evaporation * 0.5 + cfg.release_hv + 0.02
