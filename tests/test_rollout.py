from llm_mappo.lrs.rollout import evaluate_reward_fn
from llm_mappo.env.env import SearchEnv
from llm_mappo.utils.config import Config


def _zero_reward(obs, prev_obs, info):
    return 0.0


def test_evaluate_returns_finite_score():
    cfg = Config()
    env = SearchEnv(cfg)
    score = evaluate_reward_fn(_zero_reward, env, cfg, seed=0)
    assert isinstance(score, float)
    assert score >= 0.0
