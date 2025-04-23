import argparse
import datetime
import logging
from pathlib import Path
import random

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm
import wandb

from reduction_env import ReductionEnvConfig
from simulator import SimulatorTrainer


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
    dist_group.add_argument("--knapsack", action="store_true")

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
    elif args.knapsack:
        args.dist = "knapsack"

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
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
        teacher_forcing=args.teacher_forcing,
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
        for metric in metrics:
            wandb.log(metric)
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
