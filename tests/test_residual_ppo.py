"""Numerical checks for the self-contained residual PPO module."""

from __future__ import annotations

import math
import unittest

import torch
from torch.distributions import Normal

from starter.policies.residual_ppo import (
    MAX_STD,
    MIN_STD,
    PPOBatch,
    ResidualActorCritic,
    compute_gae,
    ppo_update,
)


class ComputeGAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_terminal_step_excludes_bootstrap(self):
        rewards = torch.tensor([1.0, -0.5])
        values = torch.tensor([0.5, 1.5])
        next_values = torch.tensor([2.0, 100.0])
        terminated = torch.tensor([False, True])
        episode_ends = torch.tensor([False, True])

        advantages, returns = compute_gae(
            rewards, values, next_values, terminated, episode_ends, gamma=0.9, gae_lambda=0.8
        )

        expected_adv = torch.tensor([0.86, -2.0])
        expected_returns = torch.tensor([1.36, -0.5])
        torch.testing.assert_close(advantages, expected_adv)
        torch.testing.assert_close(returns, expected_returns)

        changed, _ = compute_gae(
            rewards,
            values,
            torch.tensor([2.0, -999.0]),
            terminated,
            episode_ends,
            gamma=0.9,
            gae_lambda=0.8,
        )
        torch.testing.assert_close(changed, expected_adv)

    def test_truncation_bootstraps_but_stops_propagation(self):
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([1.0, 2.0])
        next_values = torch.tensor([3.0, 4.0])
        terminated = torch.tensor([False, False])
        episode_ends = torch.tensor([False, True])

        advantages, returns = compute_gae(
            rewards, values, next_values, terminated, episode_ends, gamma=0.9, gae_lambda=0.8
        )

        torch.testing.assert_close(advantages, torch.tensor([2.852, 1.6]))
        torch.testing.assert_close(returns, torch.tensor([3.852, 3.6]))

        terminal_advantages, _ = compute_gae(
            rewards,
            values,
            next_values,
            torch.tensor([False, True]),
            episode_ends,
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertLess(terminal_advantages[1].item(), advantages[1].item())

    def test_episode_boundaries_do_not_leak(self):
        rewards = torch.tensor([0.1, 0.2, 0.3, 0.4])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        next_values = torch.tensor([2.0, 3.0, 4.0, 5.0])
        terminated = torch.tensor([False, False, False, True])
        episode_ends = torch.tensor([False, True, False, True])

        advantages, returns = compute_gae(
            rewards, values, next_values, terminated, episode_ends, gamma=0.9, gae_lambda=0.8
        )

        changed_future, _ = compute_gae(
            rewards,
            torch.tensor([1.0, 2.0, 103.0, 104.0]),
            torch.tensor([2.0, 3.0, 104.0, 105.0]),
            terminated,
            episode_ends,
            gamma=0.9,
            gae_lambda=0.8,
        )
        torch.testing.assert_close(changed_future[:2], advantages[:2])

        changed_past, _ = compute_gae(
            torch.tensor([100.1, 100.2, 0.3, 0.4]),
            torch.tensor([101.0, 102.0, 3.0, 4.0]),
            torch.tensor([102.0, 103.0, 4.0, 5.0]),
            terminated,
            episode_ends,
            gamma=0.9,
            gae_lambda=0.8,
        )
        torch.testing.assert_close(changed_past[2:], advantages[2:])
        self.assertTrue(torch.isfinite(returns).all())


class ResidualActorCriticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0)
        self.model = ResidualActorCritic(
            feature_dim=5, action_dim=3, hidden_dim=16, initial_std=0.1
        )
        self.features = torch.randn(7, 5)

    def test_initial_mean_is_zero_and_logprob_matches_gaussian(self):
        actions, log_probs, values = self.model.act(self.features, deterministic=True)

        self.assertEqual(tuple(actions.shape), (7, 3))
        self.assertEqual(tuple(log_probs.shape), (7,))
        self.assertEqual(tuple(values.shape), (7,))
        torch.testing.assert_close(actions, torch.zeros_like(actions))

        expected = 3 * (-math.log(0.1) - 0.5 * math.log(2.0 * math.pi))
        torch.testing.assert_close(
            log_probs, torch.full_like(log_probs, expected), rtol=1e-6, atol=1e-6
        )

    def test_sampled_actions_are_scored_at_their_raw_values(self):
        actions, log_probs, values = self.model.act(self.features)
        distribution = Normal(torch.zeros_like(actions), torch.full_like(actions, 0.1))
        expected_log_probs = distribution.log_prob(actions).sum(-1)

        torch.testing.assert_close(log_probs, expected_log_probs)
        evaluated, entropy, evaluated_values = self.model.evaluate_actions(
            self.features, actions
        )
        torch.testing.assert_close(evaluated, log_probs)
        torch.testing.assert_close(evaluated_values, values)

        expected_entropy = 3 * 0.5 * math.log(2.0 * math.pi * math.e * 0.1**2)
        torch.testing.assert_close(
            entropy, torch.full_like(entropy, expected_entropy), rtol=1e-6, atol=1e-6
        )

    def test_std_is_bounded_at_both_ends(self):
        with torch.no_grad():
            self.model.log_std.fill_(math.log(10.0))
        samples, _, _ = self.model.act(torch.zeros(40_000, 5))
        sampled_std = samples.std(dim=0)
        self.assertTrue(torch.all(sampled_std <= MAX_STD + 0.01))
        torch.testing.assert_close(
            sampled_std, torch.full_like(sampled_std, MAX_STD), atol=0.02, rtol=0.0
        )

        with torch.no_grad():
            self.model.log_std.fill_(math.log(0.001))
        samples, _, _ = self.model.act(torch.zeros(40_000, 5))
        sampled_std = samples.std(dim=0)
        self.assertTrue(torch.all(sampled_std >= MIN_STD - 0.01))
        torch.testing.assert_close(
            sampled_std, torch.full_like(sampled_std, MIN_STD), atol=0.004, rtol=0.0
        )

    def test_invalid_construction_is_rejected(self):
        with self.assertRaises(ValueError):
            ResidualActorCritic(feature_dim=0)
        with self.assertRaises(ValueError):
            ResidualActorCritic(feature_dim=4, initial_std=0.0)


class PPOUpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _batch(self, model, size=64, seed=3):
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(size, model.feature_dim, generator=generator)
        with torch.no_grad():
            actions, log_probs, values = model.act(features)
            advantages = torch.randn(size, generator=generator)
            returns = torch.randn(size, generator=generator)
        return PPOBatch(
            features=features,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
        )

    def test_update_changes_actor_and_value_with_finite_metrics(self):
        torch.manual_seed(1)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model)
        actor_before = model.mean_head.weight.detach().clone()
        value_before = model.value_head.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=4,
            minibatch_size=16,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertEqual(metrics["updates"], 16)
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            self.assertTrue(math.isfinite(metrics[name]), name)
        self.assertFalse(torch.allclose(model.mean_head.weight, actor_before))
        self.assertFalse(torch.allclose(model.value_head.weight, value_before))

    def test_target_kl_stops_after_the_first_update(self):
        torch.manual_seed(2)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, seed=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=10,
            minibatch_size=64,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=1e-8,
        )

        self.assertEqual(metrics["updates"], 1)

    def test_minibatch_larger_than_batch_is_one_batch_per_epoch(self):
        torch.manual_seed(5)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=5, seed=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=2,
            minibatch_size=256,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertEqual(metrics["updates"], 2)
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            self.assertTrue(math.isfinite(metrics[name]), name)

    def test_early_stop_before_first_step_reports_finite_metrics(self):
        torch.manual_seed(7)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=8, seed=8)
        # Force a KL above threshold before any optimizer step.
        batch.log_probs = batch.log_probs + 1.0
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=3,
            minibatch_size=8,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=1e-6,
        )

        self.assertEqual(metrics["updates"], 0)
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            self.assertTrue(math.isfinite(metrics[name]), name)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], msg=name)

    def test_bad_hyperparameters_are_rejected(self):
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        with self.assertRaises(ValueError):
            ppo_update(
                model,
                optimizer,
                batch,
                update_epochs=1,
                minibatch_size=8,
                clip_range=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                target_kl=0.0,
            )


if __name__ == "__main__":
    unittest.main()
