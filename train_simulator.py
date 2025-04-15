import argparse
import datetime
import logging
from pathlib import Path
from typing import Union
import random

from fpylll import FPLLL
import numpy as np
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from modules import GSNormEncoder, ActionEncoder
from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment
from simulator import Simulator


class SimulatorTrainer(nn.Module):
    def __init__(
        self,
        env_config: ReductionEnvConfig,
        max_basis_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        super().__init__()
        self.env_config = env_config
        self.env = VectorizedReductionEnvironment(env_config)
        self.device = device

        self.gs_norm_encoder = GSNormEncoder(
            max_basis_dim=max_basis_dim, hidden_dim=hidden_dim, dropout_p=0.1
        )
        self.action_encoder = ActionEncoder(
            max_basis_dim=max_basis_dim, embedding_dim=hidden_dim
        )
        self.simulator = Simulator(
            gs_norms_encoder=self.gs_norm_encoder,
            action_encoder=self.action_encoder,
            hidden_dim=hidden_dim,
        )

        self.sim_optimizer = optim.AdamW(
            [
                *self.simulator.parameters(),
                *self.gs_norm_encoder.parameters(),
                *self.action_encoder.parameters(),
            ],
            lr=lr,
        )

        state, info = self.env.reset()
        self.state = state.to(self.device)
        self.info = info.to(self.device)

    def train(self) -> dict:
        """Performs one training step by interacting with the environment and updating the model."""
        with torch.no_grad():
            state = self._get_processed_state()
            action, continue_mask = self._generate_actions(state)
            next_state, time_taken, next_info = self._step_environment(action)

        metrics = self._compute_loss_and_update(
            state, action, continue_mask, next_state, time_taken
        )
        self._update_state(next_state, next_info)
        return metrics

    def _get_processed_state(self) -> TensorDict:
        """Preprocesses the current state and moves it to the specified device."""
        state = self.preprocess_inputs(self.state)
        return state

    def _generate_actions(self, state: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates valid actions based on the current state."""
        batch_size = state["basis_dim"].size(0)
        terminate = torch.rand(batch_size, device=self.device) < 0.2
        action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        continue_mask = ~terminate

        if continue_mask.any():
            last_actions = state["last_action"]
            basis_dims = state["basis_dim"]
            valid_continue_mask = (last_actions < basis_dims - 1) & continue_mask
            valid_indices = torch.where(valid_continue_mask)[0]

            for idx in valid_indices:
                min_action = int(last_actions[idx].item() + 1)
                max_action = basis_dims[idx].item() - 1
                action[idx] = torch.randint(
                    min_action, max_action + 1, (1,), device=self.device
                ).squeeze()

        return action, continue_mask

    def _step_environment(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, dict]:
        """Steps the environment and processes the next state."""
        next_state, _, _, _, next_info = self.env.step(action)
        return (
            self.preprocess_inputs(next_state),
            next_info["time"] - self.info["time"],
            next_info,
        )

    def _compute_loss_and_update(
        self,
        state: TensorDict,
        action: torch.Tensor,
        continue_mask: torch.Tensor,
        next_state: TensorDict,
        time_taken: torch.Tensor,
    ) -> dict:
        """Computes loss and updates model weights."""
        if not continue_mask.any():
            return {}

        # Extract relevant tensors for loss calculation
        current_gs = state["gs_norms"][continue_mask]
        prev_act = state["last_action"][continue_mask]
        basis_dim = state["basis_dim"][continue_mask]
        current_act = action[continue_mask].float()
        target_gs = next_state["gs_norms"][continue_mask]
        target_time = time_taken[continue_mask]

        # Model predictions
        predicted_gs, predicted_time, _ = self.simulator(
            current_gs_norms=current_gs,
            previous_action=prev_act,
            current_action=current_act,
            basis_dim=basis_dim,
            target_gs_norms=target_gs,
        )

        gs_loss = nn.functional.mse_loss(predicted_gs, target_gs)
        time_loss = nn.functional.mse_loss(predicted_time, target_time)
        total_loss = gs_loss + time_loss

        self.sim_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.sim_optimizer.step()

        return {
            "train/gs_loss": gs_loss.item(),
            "train/time_loss": time_loss.item(),
            "train/total_loss": total_loss.item(),
        }

    def _update_state(self, next_state: TensorDict, next_info: dict) -> None:
        """Updates the current state and environment information."""
        self.state = next_state
        self.info = TensorDict(next_info)

    @staticmethod
    def preprocess_inputs(tensordict: TensorDict) -> TensorDict:
        """Converts basis matrices into Gram-Schmidt norms for model input."""
        basis = tensordict["basis"]
        basis_dim = tensordict["basis_dim"]
        max_basis_dim = basis.size(1)

        # Create mask for valid basis dimensions
        mask = torch.arange(max_basis_dim, device=basis.device) < basis_dim.view(
            -1, 1, 1
        )
        masked_basis = basis * mask

        # Compute Gram-Schmidt norms via QR decomposition
        _, R = torch.linalg.qr(masked_basis)
        diag = torch.diagonal(R, dim1=-2, dim2=-1).abs().log()
        gs_norms = diag * mask.squeeze(1)

        return TensorDict(
            {
                "basis": basis,
                "gs_norms": gs_norms,
                "last_action": tensordict["last_action"],
                "basis_dim": basis_dim,
            },
            batch_size=[],
        )


# ---------------------- Main Execution Setup ----------------------
def parse_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="BKZ Simulator Training.")

    # Training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--chkpt-interval", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=48)

    # Environment parameters
    env_group = parser.add_argument_group("Environment Settings")
    env_group.add_argument("--train-min-dim", type=int, required=True)
    env_group.add_argument("--train-max-dim", type=int, required=True)
    env_group.add_argument("--max-basis-dim", type=int, required=True)
    env_group.add_argument("--time-limit", type=float, default=300.0)

    # Basis distribution (mutually exclusive)
    dist_group = env_group.add_mutually_exclusive_group(required=True)
    dist_group.add_argument("--uniform", action="store_true")
    dist_group.add_argument("--qary", action="store_true")
    dist_group.add_argument("--ntrulike", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validates command-line arguments."""
    if args.train_min_dim > args.train_max_dim:
        raise ValueError("train_min_dim must be <= train_max_dim")
    if args.train_max_dim > args.max_basis_dim:
        raise ValueError("train_max_dim cannot exceed max_basis_dim")


def setup_environment(args: argparse.Namespace) -> ReductionEnvConfig:
    """Creates and returns the environment configuration."""
    return ReductionEnvConfig(
        net_dim=args.max_basis_dim,
        train_min_dim=args.train_min_dim,
        train_max_dim=args.train_max_dim,
        time_limit=args.time_limit,
        distribution=args.dist,
        batch_size=args.batch_size,
        time_penalty_weight=0.0,
        defect_reward_weight=0.0,
        length_reward_weight=0.0,
    )


def initialize_logging(checkpoint_dir: Path) -> None:
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(checkpoint_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    FPLLL.set_precision(1000)
    args = parse_args()
    validate_args(args)

    # Determine distribution
    if args.uniform:
        args.dist = "uniform"
    elif args.qary:
        args.dist = "qary"
    elif args.ntrulike:
        args.dist = "ntrulike"

    # Setup reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup directories and logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_dir = Path(
        f"sim_checkpoints/{args.dist}_dim{args.max_basis_dim}_{timestamp}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initialize_logging(checkpoint_dir)

    logging.info(args)

    # Initialize training components
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('We will use the GPU: ' +
                     str(torch.cuda.get_device_name(0)))
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    env_config = setup_environment(args)

    trainer = SimulatorTrainer(
        env_config=env_config,
        max_basis_dim=args.max_basis_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
    ).to(device)

    # Initialize WandB
    wandb.init(
        project="bkz-simulator-training",
        name=f"sim_{args.dist}_dim{args.max_basis_dim}_{timestamp}",
        config=vars(args),
    )

    # Training loop
    progress_bar = tqdm(total=args.steps, desc="Training Steps")
    for step in range(args.steps):
        metrics = trainer.train()
        wandb.log(metrics)
        progress_bar.update(1)

        if (step + 1) % args.chkpt_interval == 0:
            checkpoint_path = checkpoint_dir / f"simulator_step{step+1}.pth"
            torch.save(trainer.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved at step {step+1}")

    # Final save
    final_path = checkpoint_dir / "simulator_final.pth"
    torch.save(trainer.state_dict(), final_path)
    logging.info(f"Final model saved to {final_path}")
    progress_bar.close()


if __name__ == "__main__":
    main()
