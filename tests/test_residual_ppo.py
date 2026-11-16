"""Numerical checks for the self-contained residual PPO module."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from starter.policies.residual_ppo import (
    MAX_STD,
    MIN_STD,
    PPOBatch,
    ResidualActorCritic,
    compute_gae,
    ppo_update,
)


def _grouped_optimizer(model, lr, optimizer_cls=torch.optim.Adam):
    """One optimizer with the separate actor and critic parameter groups."""

    return optimizer_cls(
        [
            {"params": model.actor_parameters(), "lr": lr},
            {"params": model.critic_parameters(), "lr": lr},
        ]
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
        optimizer = _grouped_optimizer(model, 3e-3)

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
        self.assertFalse(metrics["early_stopped"])
        self.assertEqual(metrics["epochs_completed"], 4)
        self.assertIsNotNone(metrics["grad_norm"])
        self.assertIsNotNone(metrics["max_grad_norm_observed"])
        self.assertTrue(metrics["explained_variance"] is not None)
        self.assertTrue(math.isfinite(metrics["explained_variance"]))
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "total_loss",
            "grad_norm",
            "max_grad_norm_observed",
            "policy_std_mean",
            "policy_std_min",
            "policy_std_max",
        ):
            self.assertTrue(math.isfinite(metrics[name]), name)
        self.assertFalse(torch.allclose(model.mean_head.weight, actor_before))
        self.assertFalse(torch.allclose(model.value_head.weight, value_before))

    def test_target_kl_stops_after_the_first_update(self):
        torch.manual_seed(2)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, seed=4)
        optimizer = _grouped_optimizer(model, 1e-2)

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
        self.assertTrue(metrics["early_stopped"])
        # The first epoch completed; the stop happened on the next one.
        self.assertEqual(metrics["epochs_completed"], 1)

    def test_mid_epoch_stop_excludes_incomplete_epoch(self):
        torch.manual_seed(21)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=64, seed=22)
        optimizer = _grouped_optimizer(model, 1e-2)

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
            target_kl=1e-8,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertTrue(metrics["early_stopped"])
        self.assertEqual(metrics["epochs_completed"], 0)

    def test_minibatch_larger_than_batch_is_one_batch_per_epoch(self):
        torch.manual_seed(5)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=5, seed=6)
        optimizer = _grouped_optimizer(model, 3e-3)

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
        self.assertFalse(metrics["early_stopped"])
        self.assertEqual(metrics["epochs_completed"], 2)
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "total_loss",
        ):
            self.assertTrue(math.isfinite(metrics[name]), name)

    def test_early_stop_before_first_step_reports_finite_metrics(self):
        torch.manual_seed(7)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=8, seed=8)
        # Force a KL above threshold before any optimizer step.
        batch.log_probs = batch.log_probs + 1.0
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = _grouped_optimizer(model, 3e-3)

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
        self.assertTrue(metrics["early_stopped"])
        self.assertEqual(metrics["epochs_completed"], 0)
        self.assertIsNone(metrics["grad_norm"])
        self.assertIsNone(metrics["max_grad_norm_observed"])
        self.assertTrue(math.isfinite(metrics["explained_variance"]))
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "total_loss",
            "policy_std_mean",
            "policy_std_min",
            "policy_std_max",
        ):
            self.assertTrue(math.isfinite(metrics[name]), name)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], msg=name)

    def test_explained_variance_matches_final_critic(self):
        torch.manual_seed(23)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=32, seed=24)
        optimizer = _grouped_optimizer(model, 3e-3)

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
            target_kl=10.0,
        )

        with torch.no_grad():
            _, _, final_values = model.evaluate_actions(batch.features, batch.actions)
        expected = 1.0 - torch.var(batch.returns - final_values, unbiased=False) / torch.var(
            batch.returns, unbiased=False
        )
        self.assertAlmostEqual(metrics["explained_variance"], float(expected), places=6)

    def test_explained_variance_is_none_for_constant_returns(self):
        torch.manual_seed(25)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=16, seed=26)
        batch.returns = torch.full_like(batch.returns, 0.25)
        optimizer = _grouped_optimizer(model, 3e-3)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=1,
            minibatch_size=16,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertIsNone(metrics["explained_variance"])

    def test_grad_norm_is_measured_before_clipping(self):
        torch.manual_seed(27)
        model = ResidualActorCritic(
            feature_dim=4, action_dim=3, hidden_dim=16, learn_std=True
        )
        batch = self._batch(model, size=64, seed=28)

        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        new_log_probs, entropy, new_values = model.evaluate_actions(
            batch.features, batch.actions
        )
        log_ratio = new_log_probs - batch.log_probs
        ratio = log_ratio.exp()
        policy_loss = -torch.min(
            ratio * advantages,
            ratio.clamp(0.8, 1.2) * advantages,
        ).mean()
        value_loss = F.mse_loss(new_values, batch.returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
        loss.backward()

        def _norm(parameters):
            return torch.sqrt(
                sum(
                    parameter.grad.norm() ** 2
                    for parameter in parameters
                    if parameter.grad is not None
                )
            )

        expected_actor_norm = _norm(model.actor_parameters())
        expected_critic_norm = _norm(model.critic_parameters())
        expected_norm = torch.sqrt(expected_actor_norm**2 + expected_critic_norm**2)
        model.zero_grad(set_to_none=True)
        self.assertGreater(float(expected_actor_norm), 1e-3)
        self.assertGreater(float(expected_critic_norm), 1e-3)

        optimizer = _grouped_optimizer(model, 3e-3)
        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=1,
            minibatch_size=64,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=1e-3,
            target_kl=10.0,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertAlmostEqual(
            metrics["actor_grad_norm"], float(expected_actor_norm), places=5
        )
        self.assertAlmostEqual(
            metrics["critic_grad_norm"], float(expected_critic_norm), places=5
        )
        self.assertAlmostEqual(metrics["grad_norm"], float(expected_norm), places=5)
        self.assertAlmostEqual(
            metrics["max_grad_norm_observed"], float(expected_norm), places=5
        )

    def test_frozen_std_does_not_drift(self):
        torch.manual_seed(31)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        self.assertFalse(model.log_std.requires_grad)
        self.assertFalse(any(p is model.log_std for p in model.actor_parameters()))
        self.assertFalse(any(p is model.log_std for p in model.critic_parameters()))
        self.assertIn("log_std", model.state_dict())
        std_before = model.log_std.detach().clone()

        batch = self._batch(model, size=32, seed=32)
        optimizer = _grouped_optimizer(model, 3e-2)
        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=2,
            minibatch_size=16,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=1.0,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertGreater(metrics["updates"], 0)
        torch.testing.assert_close(model.log_std, std_before)

    def test_learnable_std_receives_entropy_gradient(self):
        torch.manual_seed(33)
        model = ResidualActorCritic(
            feature_dim=4, action_dim=3, hidden_dim=16, learn_std=True
        )
        self.assertTrue(model.log_std.requires_grad)
        self.assertTrue(any(p is model.log_std for p in model.actor_parameters()))
        std_before = model.log_std.detach().clone()

        batch = self._batch(model, size=32, seed=34)
        optimizer = _grouped_optimizer(model, 3e-2)
        ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=2,
            minibatch_size=16,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=1.0,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertFalse(torch.allclose(model.log_std, std_before))

    def test_critic_only_update_does_not_change_actor_output(self):
        torch.manual_seed(35)
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=32, seed=36)
        # Zero advantages make the clipped policy surrogate exactly flat.
        batch.advantages = torch.zeros_like(batch.advantages)

        eval_features = torch.randn(17, model.feature_dim)
        with torch.no_grad():
            actor_before, _, _ = model.act(eval_features, deterministic=True)
        actor_parameters_before = [
            parameter.detach().clone() for parameter in model.actor_parameters()
        ]
        value_before = model.value_head.weight.detach().clone()

        optimizer = _grouped_optimizer(model, 1e-2)
        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=2,
            minibatch_size=8,
            clip_range=0.2,
            value_coef=1.0,
            entropy_coef=0.0,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        self.assertGreater(metrics["updates"], 0)
        with torch.no_grad():
            actor_after, _, _ = model.act(eval_features, deterministic=True)
        torch.testing.assert_close(actor_after, actor_before)
        for parameter, before in zip(model.actor_parameters(), actor_parameters_before):
            torch.testing.assert_close(parameter, before)
        self.assertFalse(torch.allclose(model.value_head.weight, value_before))

    def test_critic_return_magnitude_does_not_change_actor_step(self):
        torch.manual_seed(37)
        model_small = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        torch.manual_seed(37)
        model_huge = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)

        batch = self._batch(model_small, size=32, seed=38)
        huge_batch = PPOBatch(
            features=batch.features.clone(),
            actions=batch.actions.clone(),
            log_probs=batch.log_probs.clone(),
            values=batch.values.clone(),
            advantages=batch.advantages.clone(),
            returns=batch.returns * 1e6 + 1e4,
        )
        optimizer_small = _grouped_optimizer(model_small, 0.1, torch.optim.SGD)
        optimizer_huge = _grouped_optimizer(model_huge, 0.1, torch.optim.SGD)

        torch.manual_seed(39)
        ppo_update(
            model_small,
            optimizer_small,
            batch,
            update_epochs=1,
            minibatch_size=32,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=0.5,
            target_kl=10.0,
        )
        torch.manual_seed(39)
        ppo_update(
            model_huge,
            optimizer_huge,
            huge_batch,
            update_epochs=1,
            minibatch_size=32,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        for parameter_small, parameter_huge in zip(
            model_small.actor_parameters(), model_huge.actor_parameters()
        ):
            torch.testing.assert_close(parameter_huge, parameter_small)
        with torch.no_grad():
            actions_small, _, _ = model_small.act(batch.features, deterministic=True)
            actions_huge, _, _ = model_huge.act(batch.features, deterministic=True)
        torch.testing.assert_close(actions_huge, actions_small)
        self.assertFalse(
            torch.allclose(model_small.value_head.weight, model_huge.value_head.weight)
        )

    def test_contextual_bandit_mean_action_reward_improves(self):
        torch.manual_seed(40)
        model = ResidualActorCritic(
            feature_dim=4, action_dim=2, hidden_dim=32, initial_std=0.2
        )
        generator = torch.Generator().manual_seed(41)
        weight = torch.randn(4, 2, generator=generator) * 0.5
        eval_features = torch.randn(512, 4, generator=generator)

        def target(features):
            return torch.tanh(features @ weight)

        def mean_action_reward():
            with torch.no_grad():
                actions, _, _ = model.act(eval_features, deterministic=True)
            return float(-(actions - target(eval_features)).pow(2).sum(-1).mean())

        optimizer = _grouped_optimizer(model, 3e-3)
        before = mean_action_reward()
        for _round in range(12):
            features = torch.randn(256, 4)
            with torch.no_grad():
                actions, log_probs, values = model.act(features)
            rewards = -(actions - target(features)).pow(2).sum(-1)
            ppo_update(
                model,
                optimizer,
                PPOBatch(
                    features=features,
                    actions=actions,
                    log_probs=log_probs,
                    values=values,
                    advantages=rewards,
                    returns=rewards,
                ),
                update_epochs=4,
                minibatch_size=64,
                clip_range=0.2,
                value_coef=0.5,
                entropy_coef=0.0,
                max_grad_norm=0.5,
                target_kl=10.0,
            )
        after = mean_action_reward()

        self.assertGreater(after, before + 0.1)

    def test_policy_std_metrics_use_clamped_final_std(self):
        torch.manual_seed(29)
        model = ResidualActorCritic(
            feature_dim=4, action_dim=3, hidden_dim=16, learn_std=True
        )
        with torch.no_grad():
            model.log_std.copy_(torch.tensor([math.log(0.001), math.log(0.1), math.log(5.0)]))
        batch = self._batch(model, size=8, seed=30)
        optimizer = _grouped_optimizer(model, 3e-3)

        metrics = ppo_update(
            model,
            optimizer,
            batch,
            update_epochs=1,
            minibatch_size=8,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=10.0,
        )

        expected = model.log_std.detach().clamp(
            math.log(MIN_STD), math.log(MAX_STD)
        ).exp()
        self.assertAlmostEqual(metrics["policy_std_min"], float(expected.min()))
        self.assertAlmostEqual(metrics["policy_std_max"], float(expected.max()))
        self.assertAlmostEqual(metrics["policy_std_mean"], float(expected.mean()))
        self.assertAlmostEqual(metrics["policy_std_min"], MIN_STD)
        self.assertAlmostEqual(metrics["policy_std_max"], MAX_STD)

    def test_bad_hyperparameters_are_rejected(self):
        model = ResidualActorCritic(feature_dim=4, action_dim=3, hidden_dim=16)
        batch = self._batch(model, size=8)
        optimizer = _grouped_optimizer(model, 3e-4)
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
