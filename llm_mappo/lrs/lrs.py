import re
from llm_mappo.utils.config import Config
from llm_mappo.lrs.client import DeepSeekClient
from llm_mappo.lrs.prompt import build_initial_prompt, build_feedback_prompt
from llm_mappo.lrs.rollout import evaluate_reward_fn


def extract_code(text: str) -> str:
    """从 LLM 输出中提取 reward 函数代码块。"""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


class LRS:
    """离线 LLM 奖励塑形：初始化 + K 轮迭代反馈（Eq.22）。"""

    def __init__(self, cfg: Config, env):
        self.cfg = cfg
        self.env = env
        self.client = DeepSeekClient(cfg.lrs.model)
        self.base = build_initial_prompt(
            cfg.env.n_uav, cfg.env.grid_size, cfg.env.n_targets)
        self.buffer = []          # (code, score)

    def run(self) -> str:
        best_code, best_score = None, -1e18
        for k in range(self.cfg.lrs.n_iterations):
            prompt = self.base if k == 0 else build_feedback_prompt(
                self.base, best_code, best_score, self.buffer[-2:])
            for attempt in range(self.cfg.lrs.max_retries):
                code = extract_code(self.client.generate(prompt))
                if self._validate(code):
                    break
            else:
                continue
            score = evaluate_reward_fn(self._make_fn(code), self.env, self.cfg)
            self.buffer.append((code, score))
            if score > best_score:
                best_code, best_score = code, score
            print(f"iter {k} score={score:.2f} best={best_score:.2f}")
        return best_code

    def _validate(self, code: str) -> bool:
        return code.strip().startswith("def reward")

    def _make_fn(self, code: str):
        ns = {}
        exec(code, ns)
        return ns["reward"]
