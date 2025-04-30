from collections import defaultdict
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate

from actor_critic import ActorCritic
from modules import AuxiliaryPredictionHeads, AuxiliaryPredictorConfig
from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment


class PPOConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_grad_norm: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size

    def __str__(self):
        self_dict = vars(self)
        return f"PPOConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"


class AgentConfig:
    def __init__(
        self,
        ppo_config: PPOConfig,
        device: Union[torch.device, str],
        batch_size: int = 1,
        dropout_p: float = 0.1,
        env_config: Optional[ReductionEnvConfig] = None,
        policy_type: str = Union[
            Literal["continuous"], Literal["discrete"], Literal["joint-energ"]
        ],
        auxiliary_predictor: bool = False,
        teacher_forcing: bool = False,
        auxiliary_predictor_reward_weight: float = 0.1,
        auxiliary_predictor_config: Optional[AuxiliaryPredictorConfig] = None,
    ) -> None:
        self.ppo_config = ppo_config
        self.device = device
        self.batch_size = batch_size
        self.dropout_p = dropout_p

        self.auxiliary_predictor = auxiliary_predictor
        self.teacher_forcing = teacher_forcing

        self.policy_type = policy_type
        self.env_config = env_config if env_config is not None else ReductionEnvConfig()
        self.auxiliary_reward_weight = (
            0.0 if not auxiliary_predictor else auxiliary_predictor_reward_weight
        )
        self.auxiliary_predictor_config = (
            None
            if not auxiliary_predictor
            else (
                auxiliary_predictor_config
                if auxiliary_predictor_config is not None
                else AuxiliaryPredictorConfig()
            )
        )

    def __str__(self):
        self_dict = vars(self)
        return f"AgentConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"


class Agent(nn.Module):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.device = self.agent_config.device

        self.actor_critic = ActorCritic(
            policy_type=self.agent_config.policy_type,
            max_basis_dim=self.agent_config.env_config.net_dim,
            dropout_p=self.agent_config.dropout_p,
        )
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.agent_config.ppo_config.lr,
        )

        self.auxiliary_predictor = None
        if self.agent_config.auxiliary_predictor:
            self.auxiliary_predictor = AuxiliaryPredictionHeads(
                gs_norms_encoder=self.actor_critic.gs_norms_encoder,
                action_encoder=self.actor_critic.action_encoder,
                dropout_p=self.agent_config.dropout_p,
                hidden_dim=self.agent_config.auxiliary_predictor_config.hidden_dim,
                device=self.device,
                teacher_forcing=self.agent_config.teacher_forcing,
            )
            self.auxiliary_predictor_optimizer = optim.AdamW(
                self.auxiliary_predictor.parameters(),
                lr=self.agent_config.auxiliary_predictor_config.lr,
            )

        self.mse_loss = nn.MSELoss()

        def sample_transform(tensordict: TensorDict) -> TensorDict:
            tensordict = TensorDict(
                {key: tensordict[key].flatten(0, 1) for key in tensordict.keys()}
            )
            return tensordict

        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(),
            transform=sample_transform,
        )

        self.batch_size = self.agent_config.batch_size
        self.agent_config.env_config.batch_size = self.batch_size
        self.env = VectorizedReductionEnvironment(self.agent_config.env_config)
        self.state, self.info = self.env.reset()

        self.state = self.state.to(self.device)
        self.info = self.info.to(self.device)

    def store_transition(
        self,
        state: TensorDict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_state: TensorDict[str, torch.Tensor],
        next_info: TensorDict[str, torch.Tensor],
        current_info: TensorDict[str, torch.Tensor],
    ):
        td = TensorDict(
            {
                "state": {
                    "basis": state["basis"],
                    "last_action": state["last_action"],
                    "basis_dim": state["basis_dim"],
                },
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "done": done,
                "next_state": {
                    "basis": next_state["basis"],
                    "last_action": next_state["last_action"],
                    "basis_dim": next_state["basis_dim"],
                },
                "next_info": next_info,
                "current_info": current_info,
            },
            batch_size=[action.size(0)],
        )
        self.replay_buffer.extend(
            [*td.split(np.ones(td.batch_size[0], dtype=int).tolist(), dim=0)]
        )

    def get_action(self, state: TensorDict) -> Tuple[int, float, float, torch.Tensor]:
        with torch.no_grad():
            basis_dim = state["basis_dim"]
            last_action = state["last_action"]
            termination_prob, continue_logits, value = self.actor_critic(state)

            if self.agent_config.policy_type == "continuous":
                block_size_float, block_size_std = continue_logits.unbind(dim=1)
                if self.training:
                    terminate_dist = torch.distributions.Bernoulli(termination_prob)
                    terminate = terminate_dist.sample()
                    termination_log_probs = terminate_dist.log_prob(terminate)

                    block_size_dist = torch.distributions.Normal(
                        block_size_float,
                        block_size_std,
                    )
                    cont_block = block_size_dist.rsample()

                    with torch.no_grad():
                        # discrete block size for environment
                        discrete_block = torch.clamp(
                            torch.round(cont_block),
                            min=last_action + 1,
                            max=basis_dim,
                        )
                    # straight through estimator to preserve gradients
                    # unnecessary for PPO, but necessary for SAC
                    block_size = (
                        cont_block + (discrete_block.float() - cont_block).detach()
                    )
                    block_size_log_probs = block_size_dist.log_prob(block_size)

                else:
                    terminate = termination_prob
                    termination_log_probs = torch.log(termination_prob)

                    block_size_dist = torch.distributions.Normal(
                        block_size_float,
                        block_size_std,
                    )
                    block_size = torch.round(block_size_float)
                    block_size_log_probs = block_size_dist.log_prob(block_size_float)

                log_probs = torch.where(
                    terminate > 0.5,
                    termination_log_probs,  # only termination log prob matters
                    termination_log_probs
                    + block_size_log_probs
                    - torch.log(
                        block_size_dist.cdf(basis_dim + 0.5)
                        - block_size_dist.cdf(last_action + 0.5)  # adjust for boundary
                    ),
                )

                action = torch.where(
                    terminate > 0.5,
                    torch.tensor(0, device=block_size.device),
                    block_size - 1,
                )  # consolidate action ids

            elif (
                self.agent_config.policy_type == "discrete"
                or self.agent_config.policy_type == "joint-energy"
            ):
                terminate_dist = torch.distributions.Bernoulli(termination_prob)
                continue_dist = torch.distributions.Categorical(logits=continue_logits)
                if self.training:
                    terminate = terminate_dist.sample()
                    continue_action = continue_dist.sample()
                else:
                    terminate = (termination_prob > 0.5).float()
                    continue_action = torch.argmax(continue_logits, dim=-1)

                termination_log_probs = terminate_dist.log_prob(terminate)
                continue_log_probs = continue_dist.log_prob(continue_action)

                action = torch.where(
                    termination_prob > 0.5,
                    torch.tensor(0, device=continue_action.device),
                    continue_action + 1,
                )
                log_probs = torch.where(
                    termination_prob > 0.5,
                    termination_log_probs,  # only termination log prob matters
                    termination_log_probs + continue_log_probs,
                )

            return action, log_probs, value, termination_prob, continue_logits

    def _update_continuous(self) -> Dict[str, float]:
        self.train()

        batch = self.replay_buffer.sample(len(self.replay_buffer)).to(self.device)

        states = batch["state"]
        next_states = batch["next_state"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]
        rewards = batch["reward"]
        dones = batch["done"]

        # Calculate advantages by Generalized Advantage Estimation (GAE) using current value function
        with torch.no_grad():
            """
            This block computes the value estimates for the current states and the next states
            using the critic part of the actor-critic model.

            Using `torch.no_grad()` means we do not track gradients during this computation because
            these estimates serve as targets for calculating the advantage; they are not directly
            updated in this backward pass.

            The computed `values` and `next_values` are used to calculate the temporal-difference
            errors (deltas), which are the building blocks for GAE.
            """
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.agent_config.ppo_config.gamma,
            lmbda=self.agent_config.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        term_losses, block_losses = [], []
        term_entropies, block_entropies = [], []
        clip_fractions = []
        approx_kls = []

        if self.agent_config.auxiliary_predictor:
            auxiliary_predictor_losses_dict = defaultdict(list)

        for _ in range(self.agent_config.ppo_config.epochs):
            termination_probs, block_logits, values = self.actor_critic(states)
            block_mean_preds, block_pred_std = block_logits.unbind(dim=1)
            # term_probs, block_preds, values [batch_size]

            # Create action masks
            terminate_mask = actions == 0  # [batch_size]
            continue_mask = ~terminate_mask  # [batch_size]

            # Termination log probs (Bernoulli)
            term_dist = torch.distributions.Bernoulli(probs=termination_probs)
            term_log_probs = term_dist.log_prob(terminate_mask.float())  # [batch_size]

            # Block size log probs (Normal distribution)
            block_dist = torch.distributions.Normal(
                loc=block_mean_preds[continue_mask],
                scale=block_pred_std[continue_mask],
            )
            block_log_probs = block_dist.log_prob(
                actions[continue_mask].float()
            )  # [continue_size]

            # Combine log probs
            new_log_probs = term_log_probs
            new_log_probs[continue_mask] += block_log_probs

            r"""
            The probability ratio is
            \[
            r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
            \]
            This ratio measures how much the new policy differs from the old one.
            """
            ratios = (new_log_probs - old_log_probs).exp()

            r"""
            PPO modifies the standard policy gradient update using a clipped surrogate objective:
            \[
            L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
            \]
            """
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.agent_config.ppo_config.clip_epsilon,
                    1 + self.agent_config.ppo_config.clip_epsilon,
                )
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values, returns.squeeze(1))

            term_entropy = term_dist.entropy().mean()
            block_entropy_all = torch.zeros(actions.size(0), device=self.device)
            block_entropy_all[continue_mask] = block_dist.entropy()
            block_entropy = block_entropy_all.mean()
            # additivity property holds because termination and block size are independent
            entropy_loss = -(term_entropy + block_entropy)

            actor_critic_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            if self.agent_config.auxiliary_predictor:
                auxiliary_predictor_losses = self.get_auxiliary_predictor_loss(
                    states=states,
                    next_states=next_states,
                    actions=actions,
                    current_info=batch["current_info"],
                    next_info=batch["next_info"],
                )
                auxiliary_predictor_loss = {
                    k: v.nanmean() for k, v in auxiliary_predictor_losses.items()
                }
                auxiliary_predictor_losses = sum(auxiliary_predictor_loss.values())
                auxiliary_predictor_losses.requires_grad_()

            self.optimizer.zero_grad()
            if self.agent_config.auxiliary_predictor:
                self.auxiliary_predictor_optimizer.zero_grad()

            actor_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.agent_config.ppo_config.clip_grad_norm,
            )

            if self.agent_config.auxiliary_predictor:
                auxiliary_predictor_losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.agent_config.ppo_config.clip_grad_norm,
                )

            self.optimizer.step()
            if self.agent_config.auxiliary_predictor:
                self.auxiliary_predictor_optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(actor_critic_loss.item())

            with torch.no_grad():
                term_loss = (
                    -torch.min(surr1[terminate_mask], surr2[terminate_mask]).mean()
                    if terminate_mask.any()
                    else torch.tensor(0.0)
                )

                # Block size component loss
                block_loss = (
                    -torch.min(surr1[continue_mask], surr2[continue_mask]).mean()
                    if continue_mask.any()
                    else torch.tensor(0.0)
                )

            term_losses.append(term_loss.item())
            block_losses.append(block_loss.item())

            term_entropies.append(term_entropy.item())
            block_entropies.append(block_entropy.item())

            clipped = (ratios < 1 - self.agent_config.ppo_config.clip_epsilon) | (
                ratios > 1 + self.agent_config.ppo_config.clip_epsilon
            )
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

            if self.agent_config.auxiliary_predictor:
                for k, v in auxiliary_predictor_loss.items():
                    auxiliary_predictor_losses_dict[k].append(v.item())

        self.replay_buffer.empty()

        metrics = {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_term_loss": np.mean(term_losses),
            "update/avg_block_loss": np.mean(block_losses),
            "update/avg_term_entropy": np.mean(term_entropies),
            "update/avg_block_entropy": np.mean(block_entropies),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kls),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
            "update/clip_fraction": np.mean(clip_fractions),
        }
        if self.agent_config.auxiliary_predictor:
            for k, v in auxiliary_predictor_losses_dict.items():
                metrics[f"update/auxiliary/{k}"] = np.mean(v)

        return metrics

    def _update_discrete(self) -> Dict[str, float]:
        self.train()

        batch = self.replay_buffer.sample(len(self.replay_buffer)).to(self.device)

        states = batch["state"]
        next_states = batch["next_state"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]
        rewards = batch["reward"]
        dones = batch["done"]

        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.agent_config.ppo_config.gamma,
            lmbda=self.agent_config.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        term_losses, continue_losses = [], []
        term_entropies, continue_entropies = [], []
        clip_fractions = []
        approx_kls = []

        if self.agent_config.auxiliary_predictor:
            auxiliary_predictor_losses_dict = defaultdict(list)

        for _ in range(self.agent_config.ppo_config.epochs):
            termination_probs, continue_logits, values = self.actor_critic(states)

            # Create action masks
            terminate_mask = actions == 0  # [batch_size]
            continue_mask = ~terminate_mask  # [batch_size]

            # Termination log probs (Bernoulli)
            term_dist = torch.distributions.Bernoulli(probs=termination_probs)
            term_log_probs = term_dist.log_prob(terminate_mask.float())

            # Continue log probs (Categorical)
            continue_actions = actions[continue_mask] - 1  # adjust for 0-based index
            continue_dist = torch.distributions.Categorical(
                logits=continue_logits[continue_mask]
            )
            continue_log_probs = continue_dist.log_prob(continue_actions)

            # Combine log probs
            new_log_probs = term_log_probs
            new_log_probs[continue_mask] += continue_log_probs

            ratios = (new_log_probs - old_log_probs).exp()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.agent_config.ppo_config.clip_epsilon,
                    1 + self.agent_config.ppo_config.clip_epsilon,
                )
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values, returns.squeeze(1))

            term_entropy = term_dist.entropy().mean()

            continue_entropy_all = torch.zeros(actions.size(0), device=self.device)
            continue_entropy_all[continue_mask] = continue_dist.entropy()
            continue_entropy = continue_entropy_all.mean()
            # additivity property holds because termination and continue action are independent
            entropy_loss = -(term_entropy + continue_entropy)

            actor_critic_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            if self.agent_config.auxiliary_predictor:
                auxiliary_predictor_losses = self.get_auxiliary_predictor_loss(
                    states=states,
                    next_states=next_states,
                    actions=actions,
                    current_info=batch["current_info"],
                    next_info=batch["next_info"],
                )
                auxiliary_predictor_loss = {
                    k: v.nanmean() for k, v in auxiliary_predictor_losses.items()
                }
                auxiliary_predictor_losses = sum(auxiliary_predictor_loss.values())
                auxiliary_predictor_losses.requires_grad_()

            self.optimizer.zero_grad()
            if self.agent_config.auxiliary_predictor:
                self.auxiliary_predictor_optimizer.zero_grad()

            actor_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.agent_config.ppo_config.clip_grad_norm,
            )

            if self.agent_config.auxiliary_predictor:
                auxiliary_predictor_losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.agent_config.ppo_config.clip_grad_norm,
                )

            self.optimizer.step()
            if self.agent_config.auxiliary_predictor:
                self.auxiliary_predictor_optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(actor_critic_loss.item())

            with torch.no_grad():
                term_loss = (
                    -torch.min(surr1[terminate_mask], surr2[terminate_mask]).mean()
                    if terminate_mask.any()
                    else torch.tensor(0.0)
                )
                continue_loss = (
                    -torch.min(surr1[continue_mask], surr2[continue_mask]).mean()
                    if continue_mask.any()
                    else torch.tensor(0.0)
                )

            term_losses.append(term_loss.item())
            continue_losses.append(continue_loss.item())
            term_entropies.append(term_entropy.item())
            continue_entropies.append(continue_entropy.item())

            clipped = (ratios < 1 - self.agent_config.ppo_config.clip_epsilon) | (
                ratios > 1 + self.agent_config.ppo_config.clip_epsilon
            )
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

            if self.agent_config.auxiliary_predictor:
                for k, v in auxiliary_predictor_loss.items():
                    auxiliary_predictor_losses_dict[k].append(v.item())

        self.replay_buffer.empty()

        metrics = {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_term_loss": np.mean(term_losses),
            "update/avg_continue_loss": np.mean(continue_losses),
            "update/avg_term_entropy": np.mean(term_entropies),
            "update/avg_continue_entropy": np.mean(continue_entropies),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kls),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
            "update/clip_fraction": np.mean(clip_fractions),
        }
        if self.agent_config.auxiliary_predictor:
            for k, v in auxiliary_predictor_losses_dict.items():
                metrics[f"update/auxiliary/{k}"] = np.mean(v)

        return metrics

    def get_auxiliary_predictor_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        current_info: TensorDict[str, torch.Tensor],
        next_info: TensorDict[str, torch.Tensor],
    ) -> Dict[str, float]:
        current_features = self.actor_critic.preprocess_inputs(states)
        next_features = self.actor_critic.preprocess_inputs(next_states)

        losses = {
            "simulated_gs_norms": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "simulated_time": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "reconstructed_gs_norms": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "reconstructed_prev_action": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "reconstructed_current_action": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "reconstructed_log_defect": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
            "reconstructed_basis_dim": torch.full_like(
                actions.float(), float("nan"), device=actions.device
            ),
        }

        continue_mask = actions != 0
        if continue_mask.any():
            preds = self.auxiliary_predictor(
                current_gs_norms=current_features["gs_norms"][continue_mask],
                previous_action=current_features["last_action"][continue_mask],
                current_action=actions[continue_mask].float(),
                basis_dim=current_features["basis_dim"][continue_mask],
                target_gs_norms=(
                    current_features["gs_norms"][continue_mask]
                    if self.agent_config.teacher_forcing
                    else None
                ),
            )

            # simulation losses
            losses["simulated_gs_norms"][continue_mask] = (
                (preds["simulated_gs_norms"] - next_features["gs_norms"][continue_mask])
                ** 2
            ).mean(dim=1)
            losses["simulated_time"][continue_mask] = (
                preds["simulated_time"].exp() - next_info["time"][continue_mask].exp()
            ) ** 2

            # reconstruction losses
            losses["reconstructed_gs_norms"][continue_mask] = (
                (
                    preds["reconstructed_gs_norms"]
                    - current_features["gs_norms"][continue_mask]
                )
                ** 2
            ).mean(dim=1)
            losses["reconstructed_prev_action"][continue_mask] = (
                preds["reconstructed_prev_action"]
                - current_features["last_action"][continue_mask]
            ) ** 2
            losses["reconstructed_current_action"][continue_mask] = (
                preds["reconstructed_current_action"] - actions[continue_mask].float()
            ) ** 2
            losses["reconstructed_log_defect"][continue_mask] = (
                preds["reconstructed_log_defect"]
                - current_info["log_defect"][continue_mask].float()
            ) ** 2
            losses["reconstructed_basis_dim"][continue_mask] = (
                preds["reconstructed_basis_dim"]
                - current_features["basis_dim"][continue_mask]
            ) ** 2

        return losses

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.agent_config.ppo_config.minibatch_size:
            return {}

        if self.agent_config.policy_type == "continuous":
            metrics = self._update_continuous()
        elif (
            self.agent_config.policy_type == "discrete"
            or self.agent_config.policy_type == "joint-energy"
        ):
            # both use the same update function
            metrics = self._update_discrete()
        return metrics

    def collect_experiences(self) -> Dict[str, float]:
        with torch.no_grad():
            action, log_prob, value, terminate_prob, continue_logits = self.get_action(
                self.state
            )
            next_state, rewards, terminated, truncated, next_info = self.env.step(
                action
            )
            reward = torch.stack(list(rewards.values()), dim=0).sum(dim=0)
            if self.agent_config.auxiliary_predictor:
                auxiliary_predictor_losses = self.get_auxiliary_predictor_loss(
                    states=self.state,
                    next_states=next_state,
                    actions=action,
                    current_info=self.info,
                    next_info=next_info,
                )
                auxiliary_reward = (
                    self.agent_config.auxiliary_reward_weight
                    * torch.stack(
                        [v for v in auxiliary_predictor_losses.values()],
                        dim=0,
                    ).sum(dim=0)
                )
                auxiliary_reward_ = torch.where(action == 0, 0, auxiliary_reward)
                reward = reward + torch.clamp(auxiliary_reward_, max=10)
            done = terminated | truncated

            self.store_transition(
                state=self.state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                next_state=next_state,
                next_info=next_info,
                current_info=self.info,
            )

            metrics = []
            for i in range(reward.size(0)):
                metric = {
                    "episode/action": float(action[i]),
                    "episode/block_size": float(
                        "nan" if action[i] == 0 else action[i] + 1
                    ),
                    "episode/block_size_rel": float(
                        "nan"
                        if action[i] == 0
                        else (action[i] + 1) / self.state["basis_dim"][i]
                    ),
                    "episode/basis_dim": float(self.state["basis_dim"][i]),
                    "episode/time_taken": float(next_info["time"][i]),
                    "episode/time_penalty": float(rewards["time_penalty"][i]),
                    "episode/length_reward": float(rewards["length_reward"][i]),
                    "episode/defect_reward": float(rewards["defect_reward"][i]),
                    "episode/total_reward": float(reward[i]),
                    "episode/action_log_prob": float(log_prob[i]),
                    "episode/value_estimate": float(value[i]),
                }

                # Log actor logits based on prediction type
                metric["episode/actor_terminate_prob"] = float(terminate_prob[i])
                if action[i] != 0:
                    if self.agent_config.policy_type == "continuous":
                        metric.update(
                            {
                                "episode/actor_block_mean_logit": float(
                                    continue_logits[i][0]
                                ),
                                "episode/actor_block_std_logit": float(
                                    continue_logits[i][1]
                                ),
                            }
                        )
                    elif (
                        self.agent_config.policy_type == "discrete"
                        or self.agent_config.policy_type == "joint-energy"
                    ):
                        continue_prob = F.softmax(continue_logits[i], dim=-1)
                        continue_action = action[i] - 1  # adjust for 0-based index
                        metric["episode/actor_continue_logit"] = float(
                            continue_prob[continue_action]
                        )

                metrics.append(metric)

            if self.agent_config.auxiliary_predictor:
                for i in range(reward.size(0)):
                    metrics[i]["episode/auxiliary_reward"] = float(
                        torch.clamp(auxiliary_reward[i], max=10)
                    )
                    metrics[i]["episode/auxiliary_reward_raw"] = float(
                        auxiliary_reward[i]
                    )

                    for key in auxiliary_predictor_losses:
                        metrics[i][f"episode/{key}_loss"] = float(
                            auxiliary_predictor_losses[key][i].mean()
                        )

            self.state = next_state
            self.info = next_info

            return metrics

    def evaluate(
        self, batch: TensorDict[str, Union[torch.Tensor, TensorDict[str, torch.Tensor]]]
    ) -> Dict:
        self.eval()
        state, info = self.env.reset(options=batch)
        log_defect_history = [info["log_defect"].item()]
        shortest_length_history = [info["shortest_length"].item()]
        time_history = [info["time"].item()]

        if self.agent_config.auxiliary_predictor:
            auxiliary_predictor_losses_dict = defaultdict(list)

        done = False
        episode_reward = 0
        episode_rewards = defaultdict(int)
        steps = 0

        while not done:
            state = state.to(self.device)
            info = info.to(self.device)

            action, _, _, _, _ = self.get_action(state)
            next_state, reward, terminated, truncated, next_info = self.env.step(action)

            log_defect_history.append(next_info["log_defect"].item())
            shortest_length_history.append(next_info["shortest_length"].item())
            time_history.append(next_info["time"].item())
            done = terminated or truncated
            for key, value in reward.items():
                episode_rewards[key] += float(value)
            steps += 1

            if self.agent_config.auxiliary_predictor:
                losses_ = self.get_auxiliary_predictor_loss(
                    states=state,
                    next_states=next_state,
                    actions=action,
                    current_info=info,
                    next_info=next_info,
                )
                for k in losses_.keys():
                    auxiliary_predictor_losses_dict[k].append(float(losses_[k]))

            state = next_state
            info = next_info
        episode_reward = sum(episode_rewards.values())

        metrics = {
            "reward": episode_reward,
            "steps": steps,
            "smallest_defect": min(log_defect_history),
            "shortest_length": min(shortest_length_history),
            "tgt_length": float(batch["target_length"]),
            "success": float(
                min(shortest_length_history) < float(batch["target_length"]) + 1e-6
            ),
            "time": sum(time_history),
            "length_improvement": shortest_length_history[0]
            - shortest_length_history[-1],
            "best_length_improvement": shortest_length_history[0]
            - min(shortest_length_history),
            "gh": float(batch["gaussian_heuristic"]),
        }

        if self.agent_config.auxiliary_predictor:
            for k in auxiliary_predictor_losses_dict:
                metrics["auxiliary/" + k] = float(
                    np.nanmean(auxiliary_predictor_losses_dict[k])
                )

        metrics.update(episode_rewards)

        return metrics

    def save(self, path: Path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "agent_config": self.agent_config,
        }
        torch.save(checkpoint, path)
        return
