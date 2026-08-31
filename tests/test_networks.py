import torch
from llm_mappo.mappo.networks import Actor, Critic


def test_actor_outputs_action_probs():
    actor = Actor(obs_dim=800, n_actions=6, hidden=64)
    x = torch.randn(1, 800)
    out = actor(x)
    assert out.shape == (1, 6)
    assert torch.allclose(torch.softmax(out, dim=-1).sum(-1), torch.ones(1), atol=1e-5)


def test_critic_outputs_scalar():
    critic = Critic(obs_dim=800 * 7, hidden=64)
    x = torch.randn(1, 800 * 7)
    assert critic(x).shape == (1, 1)
