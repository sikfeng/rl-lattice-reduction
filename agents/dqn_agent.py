import numpy as np
from typing import Tuple, Optional


from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from tqdm import tqdm

from reduction_env import BKZEnvConfig, BKZEnvironment


class QNetwork(nn.Module):
    def __init__(self, basis_dim: int, action_history_size: int, action_dim: int, dropout_p: float):
        super().__init__()
        self.action_history_size = action_history_size
        self.action_dim = action_dim

        self.basis_processor = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(basis_dim ** 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256)
        )
        self.meta_processor = nn.Sequential(
            nn.Embedding(action_dim + 1, 8, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64)
        )
        self.q_net = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, action_dim)
        )

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        basis_features = self.basis_processor(tensordict["basis"])

        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        meta_features = self.meta_processor(action_history)

        combined = torch.cat([basis_features, meta_features], dim=1)
        return self.q_net(combined)


class DQNConfig:
    def __init__(self, env_config: BKZEnvConfig=None, lr=1e-4, gamma=0.99, batch_size=64, 
                 replay_buffer_size=10000, initial_epsilon=1.0, final_epsilon=0.1, 
                 epsilon_decay_steps=10000, target_update_interval=1000, dropout_p=0.2):
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.dropout_p = dropout_p
        self.env_config = env_config if env_config is not None else BKZEnvConfig()

class DQNAgent(nn.Module):
    def __init__(self, dqn_config: DQNConfig) -> None:
        super().__init__()
        self.dqn_config = dqn_config
        self.action_dim = self.dqn_config.env_config.actions_n

        self.policy_net = QNetwork(
            self.dqn_config.env_config.basis_dim,
            self.dqn_config.env_config.action_history_size,
            self.action_dim,
            dqn_config.dropout_p
        )
        self.target_net = QNetwork(
            self.dqn_config.env_config.basis_dim,
            self.dqn_config.env_config.action_history_size,
            self.action_dim,
            dqn_config.dropout_p
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=dqn_config.lr)
        
        self.replay_buffer = ReplayBuffer(storage=ListStorage(max_size=self.dqn_config.replay_buffer_size))
        self.epsilon = dqn_config.initial_epsilon
        self.epsilon_decay = (dqn_config.initial_epsilon - dqn_config.final_epsilon) / dqn_config.epsilon_decay_steps
        self.steps = 0  # Tracks number of updates for target network and epsilon decay

    def store_transition(self, state, action, reward, done, next_state):
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        td = TensorDict({
            "state": {
                "basis": state["basis"],
                "action_history": state["action_history"]
            },
            "action": action,
            "reward": reward,
            "done": done,
            "next_state": {
                "basis": next_state["basis"],
                "action_history": next_state["action_history"]
            }
        }, batch_size=[])
        self.replay_buffer.add(td)

    def get_action(self, state: TensorDict) -> Tuple[int, float, float]:
        if self.training and np.random.random() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax()

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.dqn_config.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.dqn_config.batch_size)
        states = batch["state"].flatten(0, 1)
        actions = batch["action"].flatten(0, 1)
        rewards = batch["reward"].flatten(0, 1)
        dones = batch["done"].flatten(0, 1)
        next_states = batch["next_state"].flatten(0, 1)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.dqn_config.gamma * next_q_values * (~dones)

        # Calculate loss
        loss = self.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.dqn_config.final_epsilon, 
                          self.epsilon - self.epsilon_decay)
        self.steps += 1

        # Update target network
        if self.steps % self.dqn_config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train_step(self, batch: TensorDict, device: str) -> float:
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

            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = TensorDict({
                "basis": next_state["basis"].unsqueeze(0).to(device),
                "action_history": next_state["action_history"].unsqueeze(0).to(device)
            }, batch_size=[])
            
            self.store_transition(
                state,
                action,
                reward,
                done,
                next_state
            )
            state = next_state
        
        # Update agent
        avg_reward = self.update()
        return avg_reward

    def evaluate(self, dataloader: DataLoader, device: torch.device):
        self.eval()
        env = BKZEnvironment(self.dqn_config.env_config)
        
        with torch.no_grad():
            total_reward = 0.0
            total_steps = 0
            success_count = 0
            num_samples = len(dataloader)
            
            for batch in tqdm(dataloader):
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
                    action = self.get_action(state)
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