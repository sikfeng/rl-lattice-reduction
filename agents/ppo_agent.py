import numpy as np
from typing import Dict, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from tqdm import tqdm

from reduction_env import BKZEnvConfig, BKZEnvironment


class ActorCritic(nn.Module):
    def __init__(
        self, basis_dim: int, action_history_size: int, action_dim: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_history_size = action_history_size
        
        self.basis_processor = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(basis_dim ** 2, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256)
        )
        self.meta_processor = nn.Sequential(
            nn.Embedding(action_dim + 1, 32, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64)
        )
        self.actor = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        basis_features = self.basis_processor(tensordict["basis"])

        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        meta_features = self.meta_processor(action_history)

        combined = torch.cat([basis_features, meta_features], dim=1)
        
        return self.actor(combined), self.critic(combined).squeeze(-1)


class PPOConfig:
    def __init__(self, env_config: BKZEnvConfig=None, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=4, batch_size=64, dropout_p=0.2):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.env_config = env_config if env_config is not None else BKZEnvConfig()

class PPOAgent(nn.Module):
    def __init__(self, ppo_config: PPOConfig) -> None:
        super().__init__()
        self.ppo_config = ppo_config
        self.action_dim = self.ppo_config.env_config.actions_n

        self.actor_critic = ActorCritic(
            self.ppo_config.env_config.basis_dim,
            self.ppo_config.env_config.action_history_size,
            self.action_dim,
            ppo_config.dropout_p
        )

        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=ppo_config.lr)
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage()  # Dynamically grows
        )
    
    def store_transition(self, state, action, log_prob, reward, done, next_state):        
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        td = TensorDict({
            "state": {
                "basis": state["basis"],
                "action_history": state["action_history"]
            },
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "done": done,
            "next_state": {
                "basis": next_state["basis"],
                "action_history": next_state["action_history"]
            }
        }, batch_size=[])
        self.replay_buffer.add(td)

    def get_action(self, state: TensorDict) -> Tuple[int, float, float]:
        with torch.no_grad():
            logits, value = self.actor_critic(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action, dist.log_prob(action), value

    def update(self, device: Union[torch.device, str]) -> None:
        if len(self.replay_buffer) < self.ppo_config.batch_size:
            return None
        
        # Is there a nicer way to do the following?
        batch = self.replay_buffer.sample(len(self.replay_buffer)).to(device)
        states = batch["state"].flatten(0, 1)
        next_states = batch["next_state"].flatten(0, 1)
        actions = batch["action"].flatten(0, 1)
        old_log_probs = batch["log_prob"].flatten(0, 1)
        rewards = batch["reward"].flatten(0, 1)
        dones = batch["done"].flatten(0, 1)

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
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
        

        """
        Calculates differences in values estimates with Bellman Equation. GAE builds on these
        errors by summing them over future time steps in a weighted manner.

        Multiplying next_values by (1 - dones) ensures that if a state is terminal (done == 1),
        the value contribution from the next state is zero.
        """
        deltas = rewards + self.ppo_config.gamma * next_values * (~dones) - values

        """
        The loop processes time steps in reverse order, starting from the last timestep. This is
        key because the advantage at a given timestep depends on the advantage of the following timestep.

        This loop implements the GAE formula by recursively accumulating discounted future TD errors 
        to compute a more stable advantage estimate for each timestep.
        """
        advantages = torch.zeros_like(deltas)
        advantages[-1] = deltas[-1]
        for t in reversed(range(len(deltas)-1)):
            advantages[t] = deltas[t] + self.ppo_config.gamma * self.ppo_config.gae_lambda \
                * (~dones[t]) * advantages[t + 1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Should be using old values from the rollout for the return computation
        returns = advantages + values

        # Policy updates
        for _ in range(self.ppo_config.epochs):
            logits, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            
            r"""
            The probability ratio is 
            \[ r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \]
            This ratio measures how much the new policy differs from the old one.
            """
            ratios = (new_log_probs - old_log_probs).exp()
            
            r"""
            PPO modifies the standard policy gradient update using a clipped surrogate objective:
            \[ L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right] \]
            """
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_config.clip_epsilon, 
                              1 + self.ppo_config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = self.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

        avg_reward = rewards.mean().item()
        self.replay_buffer.empty()
        return avg_reward

    def train_step(self, batch: TensorDict, device: Union[torch.device, str]) -> float:
        self.train()

        # Unpack batch data
        basis = batch['basis'] # [batch_size, n_dim, n_dim]
        shortest_vector = batch['shortest_vector'] # [batch_size, n_dim]
        
        # Reset environment
        env = BKZEnvironment(self.ppo_config.env_config)
        state, _ = env.reset(options={'basis': basis.squeeze(), 'shortest_vector': shortest_vector.squeeze()})
        done = False

        state = TensorDict({
            "basis": state["basis"].unsqueeze(0).to(device),
            "action_history": state["action_history"].unsqueeze(0).to(device)
        }, batch_size=[])
        
        # Run episode
        while not done: # should use step count as max

            action, log_prob, _ = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = TensorDict({
                "basis": next_state["basis"].unsqueeze(0).to(device),
                "action_history": next_state["action_history"].unsqueeze(0).to(device)
            }, batch_size=[])
            
            self.store_transition(
                state,
                action,
                log_prob,
                reward,
                done,
                next_state
            )
            state = next_state
        
        # Update agent
        avg_reward = self.update(device)
        return avg_reward

    def evaluate(self, dataloader: DataLoader, device: Union[torch.device, str]) -> Dict[str, float]:
        self.eval()
        env = BKZEnvironment(self.ppo_config.env_config)
        
        with torch.no_grad():
            total_reward = 0.0
            total_steps = 0
            success_count = 0
            num_samples = len(dataloader)
            
            for batch in tqdm(dataloader, dynamic_ncols=True):
                # Unpack batch data
                basis = batch['basis'].squeeze()
                shortest_vector = batch['shortest_vector'].squeeze()
                
                state, info = env.reset(options={'basis': basis, 'shortest_vector': shortest_vector})
                log_defect_history = [info["log_defect"]]
                done = False
                episode_reward = 0
                steps = 0
                
                state = TensorDict({
                    "basis": state["basis"].unsqueeze(0).to(device),
                    "action_history": state["action_history"].unsqueeze(0).to(device)
                }, batch_size=[])
                
                while not done:
                    action, _, _ = self.get_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = TensorDict({
                        "basis": next_state["basis"].unsqueeze(0).to(device),
                        "action_history": next_state["action_history"].unsqueeze(0).to(device)
                    }, batch_size=[])

                    log_defect_history.append(info["log_defect"])
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                    state = next_state
                
                total_reward += episode_reward
                total_steps += steps
                
                # Check success
                final_log_defect = log_defect_history[-1]
                if final_log_defect - batch["lll_log_defect"] < 1e-3:
                    success_count += 1
            
            return {
                'avg_reward': total_reward / num_samples,
                'avg_steps': total_steps / num_samples,
                'success_rate': success_count / num_samples
            }