from typing import List, Tuple

from llm_mappo.utils.config import EnvConfig

_cfg = EnvConfig()


def sensing_radius_grids(altitude_level: int) -> int:
    """表 I：感知域半径（网格数）随高度档位变化。"""
    return _cfg.sensing_radius_grids[altitude_level]


def detection_prob(altitude_level: int) -> float:
    return _cfg.detect_probs[altitude_level]


def false_alarm_prob(altitude_level: int) -> float:
    return _cfg.false_alarm_probs[altitude_level]


def sensing_domain(
    uav_pos: Tuple[int, int, int], radius: int, grid_size: int
) -> List[Tuple[int, int]]:
    """Eq.(4)：以 (x,y) 为中心、radius 为半径的网格集合（含边界裁剪）。"""
    x, y, _ = uav_pos
    dom = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < grid_size and 0 <= yy < grid_size:
                dom.append((xx, yy))
    return dom
