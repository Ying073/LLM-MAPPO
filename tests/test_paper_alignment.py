import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import torch

from reproduction.algorithms.networks import Actor
from reproduction.algorithms.mappo import MAPPO
from reproduction.algorithms.buffer import RolloutBuffer
from reproduction.env.search_env import MAX_STEPS, N_TARGET, N_UAV, SearchEnv
from reproduction.env.batched_search_env import BatchedSearchEnv
from reproduction.env.env_wrapper import MultiAgentWrapper
from reproduction.algorithms.dpes import PheromoneMap


class SearchEnvironmentPaperAlignmentTests(unittest.TestCase):
    def test_dynamic_target_population_is_preserved(self):
        env = SearchEnv(seed=7)
        actions = np.zeros(N_UAV, dtype=np.int64)
        for _ in range(MAX_STEPS):
            env.step(actions)
            self.assertEqual(int(env.zeta.sum()), N_TARGET)

    def test_target_speed_is_one_meter_per_second(self):
        env = SearchEnv(seed=17)
        env.occ.fill(0)
        env.target_xy_m = np.asarray(
            [[50.0 + 100.0 * (k % 10), 50.0 + 100.0 * (k // 10)] for k in range(N_TARGET)],
            dtype=np.float64,
        )
        env.target_heading[:] = 0.0
        env._refresh_zeta()
        before = env.target_xy_m.copy()
        env._step_targets()
        displacement = np.linalg.norm(env.target_xy_m - before, axis=1)
        np.testing.assert_allclose(displacement, np.ones(N_TARGET), atol=1e-9)

    def test_batched_dynamic_target_population_is_preserved(self):
        env = BatchedSearchEnv(n_envs=3, base_seed=19)
        actions = np.zeros((3, N_UAV), dtype=np.int64)
        for _ in range(50):
            env.step(actions)
            np.testing.assert_array_equal(env.zeta.sum(axis=(1, 2)), np.full(3, N_TARGET))

    def test_batched_reset_restores_all_belief_maps(self):
        env = BatchedSearchEnv(n_envs=2, base_seed=3)
        env.step(np.zeros((2, N_UAV), dtype=np.int64))
        env.reset()
        self.assertTrue(np.all(env.ltpm == np.float32(0.5)))
        self.assertTrue(np.all(env.leum == np.float32(1.0)))
        self.assertTrue(np.all(env.gtpm == np.float32(0.5)))
        self.assertTrue(np.all(env.geum == np.float32(1.0)))

    def test_action_mask_rejects_bounds_obstacles_and_uav_collisions(self):
        wrapper = MultiAgentWrapper(seed=11, use_dpes=False)
        wrapper.reset()
        wrapper.env.occ.fill(0)
        wrapper.env.uav_pos[0] = [0, 0, 1]
        wrapper.env.uav_pos[1] = [1, 0, 1]
        masks = wrapper.get_action_masks()
        self.assertFalse(bool(masks[0, 0]))  # north leaves map
        self.assertFalse(bool(masks[0, 1]))  # east enters UAV 1 position
        wrapper.env.occ[1, 0] = 1
        wrapper.env.obs_h[1, 0] = 2
        masks = wrapper.get_action_masks()
        self.assertFalse(bool(masks[0, 2]))  # south enters obstacle

    def test_observation_contains_all_uav_positions(self):
        wrapper = MultiAgentWrapper(seed=13, use_dpes=False)
        obs = wrapper.reset()[0]
        expected_position_prefix = wrapper.env.uav_pos.astype(np.float32).copy()
        expected_position_prefix[:, 0] /= 19.0
        expected_position_prefix[:, 1] /= 19.0
        expected_position_prefix[:, 2] /= 2.0
        np.testing.assert_allclose(obs[: N_UAV * 3], expected_position_prefix.reshape(-1))

    def test_paper_reward_mode_returns_equation_34_team_reward(self):
        from reproduction.reward.paper_reward import compute_paper_rbest

        wrapper = MultiAgentWrapper(seed=29, use_dpes=False, use_paper_reward=True)
        wrapper.reset()
        prev_geum = wrapper.env.geum.copy()
        actions = np.argmax(wrapper.get_action_masks(), axis=1)
        _, shared, _, info = wrapper.step(actions)
        expected, _ = compute_paper_rbest(prev_geum, wrapper.env, actions)
        self.assertAlmostEqual(shared, expected, places=5)
        np.testing.assert_allclose(info["per_agent_reward"], np.full(N_UAV, expected))

    def test_batched_wrapper_exposes_paper_observation_and_masks(self):
        from reproduction.env.batched_env_wrapper import BatchedMultiAgentWrapper

        wrapper = BatchedMultiAgentWrapper(n_envs=2, base_seed=31, use_dpes=True,
                                           use_paper_reward=True)
        obs = wrapper.reset()
        self.assertEqual(obs.shape, (2, N_UAV, 96))
        masks = wrapper.get_action_masks()
        self.assertEqual(masks.shape, (2, N_UAV, 6))
        self.assertTrue(masks.any(axis=-1).all())

    def test_energy_uses_paper_propulsion_model(self):
        from reproduction.env.search_env import E_INITIAL, TIME_STEP, UAV_SPEED, propulsion_power

        wrapper = MultiAgentWrapper(seed=43, use_dpes=False)
        wrapper.reset()
        before = wrapper.energy.copy()
        actions = np.argmax(wrapper.get_action_masks(), axis=1)
        wrapper.step(actions)
        expected_drop = propulsion_power(UAV_SPEED) * TIME_STEP / E_INITIAL
        np.testing.assert_allclose(before - wrapper.energy, np.full(N_UAV, expected_drop),
                                   rtol=2e-5, atol=1e-8)


class MAPPONetworkPaperAlignmentTests(unittest.TestCase):
    def test_actor_uses_two_relu_hidden_layers(self):
        actor = Actor(obs_dim=8, act_dim=6, hidden=64)
        relus = [m for m in actor.modules() if isinstance(m, torch.nn.ReLU)]
        tanhs = [m for m in actor.modules() if isinstance(m, torch.nn.Tanh)]
        self.assertEqual(len(relus), 2)
        self.assertEqual(len(tanhs), 0)

    def test_critic_uses_two_64_unit_relu_hidden_layers(self):
        from reproduction.algorithms.networks import Critic

        critic = Critic(global_state_dim=56)
        linear = [m for m in critic.net if isinstance(m, torch.nn.Linear)]
        relus = [m for m in critic.net if isinstance(m, torch.nn.ReLU)]
        self.assertEqual([m.out_features for m in linear[:-1]], [64, 64])
        self.assertEqual(len(relus), 2)

    def test_masked_actor_never_samples_an_invalid_action(self):
        actor = Actor(obs_dim=8, act_dim=6, hidden=64)
        obs = torch.zeros((128, 8))
        mask = torch.zeros((128, 6), dtype=torch.bool)
        mask[:, 3] = True
        actions, _, _ = actor.get_action(obs, mask)
        self.assertTrue(torch.all(actions == 3))

    def test_mappo_has_one_actor_per_uav_and_paper_hyperparameters(self):
        algo = MAPPO(obs_dim=8, global_dim=56, n_agents=N_UAV, device="cpu")
        self.assertEqual(len(algo.actors), N_UAV)
        self.assertEqual(len({id(a) for a in algo.actors}), N_UAV)
        self.assertAlmostEqual(algo.gamma, 0.95)
        self.assertAlmostEqual(algo.actor_opts[0].param_groups[0]["lr"], 0.0002)
        self.assertAlmostEqual(algo.critic_opt.param_groups[0]["lr"], 0.0002)

    def test_mappo_update_accepts_masked_multi_actor_rollout(self):
        algo = MAPPO(obs_dim=8, global_dim=56, n_agents=N_UAV, device="cpu",
                     update_epochs=1, minibatch_size=4)
        buffer = RolloutBuffer(rollout_len=2, n_uav=N_UAV, obs_dim=8,
                               global_dim=56, n_envs=1)
        masks = np.ones((N_UAV, 6), dtype=bool)
        for t in range(2):
            obs = np.full((N_UAV, 8), t, dtype=np.float32)
            actions, logp = algo.select_actions(obs, masks)
            buffer.store(obs, np.full(56, t, dtype=np.float32), actions, logp,
                         np.ones(N_UAV, dtype=np.float32), t == 1, 0.0,
                         action_mask=masks)
        actor_loss, critic_loss = algo.update(buffer, last_value=0.0)
        self.assertTrue(np.isfinite(actor_loss))
        self.assertTrue(np.isfinite(critic_loss))

    def test_checkpoint_round_trip_preserves_deterministic_policy(self):
        algo = MAPPO(obs_dim=8, global_dim=56, n_agents=N_UAV, device="cpu")
        obs = np.arange(N_UAV * 8, dtype=np.float32).reshape(N_UAV, 8) / 10.0
        masks = np.ones((N_UAV, 6), dtype=bool)
        expected, _ = algo.select_actions(obs, masks, deterministic=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            algo.save_checkpoint(path)
            restored = MAPPO.from_checkpoint(path, device="cpu")
            actual, _ = restored.select_actions(obs, masks, deterministic=True)
        np.testing.assert_array_equal(actual, expected)


class PaperRewardTests(unittest.TestCase):
    def test_equation_34_components_use_transition_not_replayed_action(self):
        from reproduction.reward.paper_reward import paper_reward_components

        env = SearchEnv(seed=23)
        env.geum.fill(1.0)
        env.geum[0, 0] = 0.2
        env.searched.fill(0)
        env.searched[0, 0] = 1
        env.searched[1, 1] = 1
        prev_geum = np.ones_like(env.geum)
        actions = np.zeros(N_UAV, dtype=np.int64)
        parts = paper_reward_components(prev_geum, env, actions)
        self.assertEqual(parts["n_searched"], 2.0)
        self.assertEqual(parts["n_uncertainty_crossings"], 1.0)
        self.assertAlmostEqual(parts["delta_uncertainty"], 0.8, places=5)

    def test_lrs_candidate_objective_accumulates_search_indicators_over_time(self):
        from reproduction.lrs import evaluate_candidate

        class FakeEnv:
            def __init__(self):
                self.i = 0

            def reset(self):
                self.i = 0

            def area_uncertainty(self):
                return 1.0 if self.i == 0 else 0.2

            def step(self, actions):
                self.i += 1
                searched = self.i
                return None, 0.0, self.i == 2, {
                    "searched_count": searched,
                    "area_uncertainty": 0.2,
                }

        with patch("reproduction.lrs.greedy_step", return_value=[0] * N_UAV):
            score, metrics = evaluate_candidate(lambda *args: 0.0, FakeEnv(), seed=0)
        self.assertAlmostEqual(score, 3.0 - 0.2)
        self.assertEqual(metrics["cumulative_searched"], 3)


class DPESPaperAlignmentTests(unittest.TestCase):
    def test_high_value_update_matches_equations_14_and_15(self):
        env = SearchEnv(seed=37)
        env.gtpm.fill(0.5)
        env.t = 1
        env.t_last_visit.fill(1)
        env.uav_pos[:] = np.asarray([[10, 10, 1], [1, 1, 1], [2, 2, 1],
                                     [3, 3, 1], [4, 4, 1], [5, 5, 1], [6, 6, 1]])
        env.gtpm[10, 10] = 0.6
        env.gtpm[9, 10] = 0.6
        pheromone = PheromoneMap()
        pheromone.dp[9, 10] = 1.0
        pheromone.update(env)
        # (1-E) * ((1-G) * (old + release) + incoming diffusion)
        expected = 0.9 * (0.9 * 0.1 + 0.1 / 4.0)
        self.assertAlmostEqual(float(pheromone.dp[10, 10]), expected, places=6)

    def test_long_unvisited_release_does_not_diffuse_to_neighbours(self):
        env = SearchEnv(seed=39)
        env.gtpm.fill(0.5)
        env.gtpm[10, 10] = 0.6
        env.t = 201
        env.t_last_visit.fill(201)
        env.t_last_visit[9, 10] = 0
        env.uav_pos[:] = np.asarray([[10, 9, 1], [1, 1, 1], [2, 2, 1],
                                     [3, 3, 1], [4, 4, 1], [5, 5, 1], [6, 6, 1]])
        pheromone = PheromoneMap()
        pheromone.dp[9, 10] = 1.0
        pheromone.update(env)
        self.assertAlmostEqual(float(pheromone.dp[10, 10]), 0.0, places=7)

    def test_pheromone_patch_is_limited_to_sensing_domain(self):
        env = SearchEnv(seed=41)
        env.uav_pos[0] = [10, 10, 0]
        pheromone = PheromoneMap()
        pheromone.dp.fill(1.0)
        patch = pheromone.get_patch(env, 0, half=2)
        self.assertEqual(float(patch.sum()), 1.0)

    def test_batched_dpes_matches_single_environment_at_boundaries(self):
        from reproduction.algorithms.batched_dpes import BatchedPheromoneMap

        single_env = SearchEnv(seed=51)
        batch_env = BatchedSearchEnv(n_envs=1, base_seed=51)
        single_env.gtpm.fill(0.6)
        single_env.t = 1
        single_env.t_last_visit.fill(1)
        batch_env.gtpm[0] = single_env.gtpm
        batch_env.t[0] = single_env.t
        batch_env.t_last[0] = single_env.t_last_visit
        batch_env.uav_pos[0] = single_env.uav_pos
        initial = np.arange(400, dtype=np.float32).reshape(20, 20) / 400.0
        single = PheromoneMap()
        batched = BatchedPheromoneMap(1)
        single.dp[:] = initial
        batched.dp[0] = initial
        single.update(single_env)
        batched.update(batch_env)
        np.testing.assert_allclose(batched.dp[0], single.dp, rtol=1e-6, atol=1e-7)


class BatchedTrainingProtocolTests(unittest.TestCase):
    def test_requested_episode_count_rounds_up(self):
        from reproduction.train_batched import compute_n_outer

        self.assertEqual(compute_n_outer(3000, 16), 188)
        self.assertEqual(compute_n_outer(100, 16), 7)

    def test_evaluation_is_learning_free_and_reports_paper_metrics(self):
        from reproduction.evaluate import evaluate_seed

        env = MultiAgentWrapper(seed=47, use_dpes=True, use_paper_reward=True)
        obs_dim = env.reset()[0].shape[0]
        algo = MAPPO(obs_dim=obs_dim, global_dim=obs_dim * N_UAV,
                     n_agents=N_UAV, device="cpu")
        before = [p.detach().clone() for actor in algo.actors for p in actor.parameters()]
        metrics = evaluate_seed(algo, seed=47, max_steps=3, use_dpes=True)
        after = [p.detach().clone() for actor in algo.actors for p in actor.parameters()]
        self.assertEqual(metrics["seed"], 47)
        self.assertIn("success_rate", metrics)
        self.assertIn("target_search_time", metrics)
        self.assertIn("terminal_area_uncertainty", metrics)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
