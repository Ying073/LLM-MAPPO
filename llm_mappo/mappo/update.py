import torch
import torch.nn.functional as F


def ppo_actor_loss(actor, obs, old_logits, actions, adv, clip_eps: float):
    """Eq.(23)(24)：PPO 裁剪替代目标（返回负值作为损失）。"""
    new_logits = actor(obs)
    old_logp = F.log_softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    new_logp = F.log_softmax(new_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()
