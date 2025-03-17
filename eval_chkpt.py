import argparse
import logging
from pathlib import Path
import random

from fpylll import FPLLL
import numpy as np
import torch

from ppo_agent import PPOAgent, PPOConfig
from load_dataset import load_lattice_dataloaders
from reduction_env import ReductionEnvConfig


def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chkpt", "--checkpoint", type=str)
    parser.add_argument("-d", "--dim", type=int, default=4)
    parser.add_argument("--distribution", type=str, default="uniform", choices=["uniform", "qary", "ntrulike"])
    parser.add_argument("--min_block_size", type=int, default=2)
    parser.add_argument("--max_block_size", type=int)
    parser.add_argument("--time-limit", type=float, default=300.0)
    args = parser.parse_args()

    # Set default for max_block_size
    if args.max_block_size is None:
        args.max_block_size = args.dim

    # Validation
    if args.min_block_size < 2:
        raise ValueError("min_block_size must be at least 2.")
    if args.max_block_size > args.dim:
        raise ValueError("max_block_size must be at most dim.")
    if args.min_block_size > args.max_block_size:
        raise ValueError("min_block_size cannot be greater than max_block_size.")

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('We will use the GPU: ' + str(torch.cuda.get_device_name(0)))
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    data_dir = Path("random_bases")

    # Create DataLoaders
    _, val_loader, test_loader = load_lattice_dataloaders(
        data_dir=data_dir,
        dimension=args.dim,
        distribution_type=args.distribution,
        batch_size=1,
        shuffle=True,
        device=device
    )

    # Environment and agent configuration
    env_config = ReductionEnvConfig(
        basis_dim=args.dim,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        batch_size=1,
        time_limit=args.time_limit
    )

    ppo_config = PPOConfig(env_config=env_config)
    best_agent = PPOAgent(ppo_config=ppo_config).to(device)

    best_agent.load_state_dict(torch.load(args.chkpt))
    total_params = sum(p.numel() for p in best_agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    val_metrics = best_agent.evaluate(val_loader, device)
    logging.info(f"Val Success: {val_metrics['success_rate']:.2f}, Avg Reward: {val_metrics['avg_reward']}, Avg Steps: {val_metrics['avg_steps']}")

    test_metrics = best_agent.evaluate(test_loader, device)
    logging.info(f"Test Success: {test_metrics['success_rate']:.2f}, Avg Reward: {test_metrics['avg_reward']}, Avg Steps: {test_metrics['avg_steps']}")

if __name__ == "__main__":
    main()
