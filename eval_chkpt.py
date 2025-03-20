import argparse
import logging
from pathlib import Path
import random
import yaml

from fpylll import FPLLL
import numpy as np
import torch

from ppo_agent import PPOAgent, PPOConfig
from load_dataset import load_lattice_dataloaders
from reduction_env import ReductionEnvConfig


def eval_model(agent, val_loader, test_loader, device):
    val_metrics = agent.evaluate(val_loader, device)
    logging.info(f"Validation metrics:")
    logging.info(str(val_metrics))

    test_metrics = agent.evaluate(test_loader, device)
    logging.info(f"Test metrics:")
    logging.info(str(test_metrics))

    metrics = {
        "val": val_metrics,
        "test": test_metrics
    }

    return metrics


def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chkpt", "--checkpoint", type=str)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--distribution", type=str,
                        default="uniform", choices=["uniform", "qary", "ntrulike"])
    parser.add_argument("--max_block_size", type=int)
    parser.add_argument("--time-penalty-weight", type=float, default=-1.0)
    parser.add_argument("--defect-reward-weight", type=float, default=0.1)
    parser.add_argument("--length-reward-weight", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=300.0)
    args = parser.parse_args()

    # Set default for max_block_size
    if args.max_block_size is None:
        args.max_block_size = args.dim

    # Validation
    if args.max_block_size > args.dim:
        raise ValueError("max_block_size must be at most dim.")
    if 2 > args.max_block_size:
        raise ValueError("max_block_size cannot be less than 2.")

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
        logging.info(
            f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('We will use the GPU: ' +
                     str(torch.cuda.get_device_name(0)))
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
        max_block_size=args.max_block_size,
        time_penalty_weight=args.time_penalty_weight,
        defect_reward_weight=args.defect_reward_weight,
        length_reward_weight=args.length_reward_weight,
        time_limit=args.time_limit
    )

    ppo_config = PPOConfig(env_config=env_config)
    agent = PPOAgent(ppo_config=ppo_config).to(device)
    if args.chkpt is not None:
        agent.load_state_dict(torch.load(args.chkpt))

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    metrics = eval_model(agent, val_loader, test_loader, device)
    logging.info(f"Validation metrics:")
    logging.info(str(metrics["val"]))
    logging.info(f"Test metrics:")
    logging.info(str(metrics["test"]))

    if args.chkpt is not None:
        yaml_file = Path(args.chkpt).with_suffix(".yaml")
        with open(yaml_file, "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        logging.info(f"Saved metrics to {yaml_file}")


if __name__ == "__main__":
    main()
