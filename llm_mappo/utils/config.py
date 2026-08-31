from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    grid_size: int = 20
    cell_size: float = 100.0          # m
    n_uav: int = 7
    n_obstacles: int = 20
    n_targets: int = 15
    altitudes: tuple = (50.0, 100.0, 150.0)
    detect_probs: tuple = (0.9, 0.8, 0.7)
    false_alarm_probs: tuple = (0.1, 0.2, 0.3)
    sensing_radius_grids: tuple = (1, 5, 9)   # 表 I「感知域大小 1/5/9」，解释为半径（网格）
    target_speed: float = 1.0          # m/s
    dt: float = 1.0                    # s
    uav_speed: float = 10.0            # m/s
    max_steps: int = 500
    # 能耗参数 (论文 p.11)
    P_b: float = 79.86
    P_d: float = 88.63
    U_tip: float = 120.0
    v0: float = 4.03
    drag: float = 0.6
    rho: float = 1.225
    solidity: float = 0.05
    area: float = 0.503
    E_ini: float = 6e4
    # 目标搜索确认阈值 (Eq.11)
    xi: float = 0.9


@dataclass
class MAPPOConfig:
    lr: float = 2e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    hidden_size: int = 64
    n_episodes: int = 28000
    entropy_coef: float = 0.01


@dataclass
class DPESConfig:
    revisit_threshold: int = 200        # D
    evaporation: float = 0.1            # E_s
    diffusion: float = 0.1              # G_s
    release_hv: float = 0.1             # 高价值释放量
    release_lu: float = 0.003           # 长期未访释放量
    release_cs: float = 0.2             # 已确认释放量
    p_hv_low: float = 0.3
    p_hv_high: float = 0.9


@dataclass
class LRSConfig:
    n_iterations: int = 5               # K
    model: str = "deepseek-reasoner"
    max_retries: int = 3


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    dpes: DPESConfig = field(default_factory=DPESConfig)
    lrs: LRSConfig = field(default_factory=LRSConfig)
    seed: int = 0
