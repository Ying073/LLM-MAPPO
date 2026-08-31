import torch
from llm_mappo.mappo.networks import Actor
from llm_mappo.mappo.update import ppo_actor_loss


def test_ppo_loss_finite_and_grad():
    actor = Actor(obs_dim=4, n_actions=3, hidden=16)
    obs = torch.randn(5, 4)
    old_logits = torch.randn(5, 3)
    actions = torch.randint(0, 3, (5,))
    adv = torch.randn(5)
    loss = ppo_actor_loss(actor, obs, old_logits, actions, adv, clip_eps=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert actor.net[0].weight.grad is not None
