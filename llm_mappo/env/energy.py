import numpy as np
from llm_mappo.utils.config import EnvConfig

_cfg = EnvConfig()


def propulsive_power(v: float) -> float:
    """Eq.(1)：旋翼 UAV 推进功率（桨叶剖面 + 诱导 + 寄生阻力）。"""
    term1 = _cfg.P_b * (1.0 + 3.0 * v ** 2 / _cfg.U_tip ** 2)
    term2 = _cfg.P_d * np.sqrt(
        np.sqrt(1.0 + v ** 4 / (4.0 * _cfg.v0 ** 4)) - v ** 2 / (2.0 * _cfg.v0 ** 2)
    )
    term3 = 0.5 * _cfg.drag * _cfg.rho * _cfg.solidity * _cfg.area * v ** 3
    return term1 + term2 + term3
