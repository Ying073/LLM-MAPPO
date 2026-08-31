from llm_mappo.utils.config import DPESConfig


def classify_grid(p: float, last_visit: int, t: int, cfg: DPESConfig) -> str:
    """Eq.(13)：按目标概率与最近访问时间把网格分四类。"""
    if p >= cfg.p_hv_high:
        return "cs"          # 高置信度，状态已确认
    if p >= cfg.p_hv_low:
        return "hv"          # 中高概率，高价值
    if t - last_visit > cfg.revisit_threshold:
        return "lu"          # 长期未访
    return "other"


def update_pheromone(category: str, dp: float, released: float,
                     diffused: float, visited: bool, cfg: DPESConfig) -> float:
    """Eq.(14)-(17)：四类信息素更新规则。"""
    if category == "hv":
        return cfg.evaporation * dp + released + diffused        # Eq.(14)
    if category == "lu":
        return 0.0 if visited else released                      # Eq.(16) 访问即清零
    if category == "cs":
        # Eq.(17)：蒸发；超过 D 由 classify 转为其他类，这里仅做蒸发
        return cfg.evaporation * dp
    return dp
