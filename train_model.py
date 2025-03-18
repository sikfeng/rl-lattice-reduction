import argparse
import datetime
import logging
from pathlib import Path
import random

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm

from ppo_agent import PPOAgent, PPOConfig
from load_dataset import load_lattice_dataloaders
from reduction_env import ReductionEnvConfig


def main():
    distributions = ["uniform", "qary", "ntrulike"]

    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("-d", "--dim", type=int, default=4)
    parser.add_argument("--distribution", type=str,
                        choices=distributions)
    parser.add_argument("--max-block-size", type=int)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_dir = Path(
        f"checkpoint/ppo-model_dim-{args.dim}_{start_timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(checkpoint_dir / "training.log", mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info(args)

    logging.info(f"Saving run to checkpoint directory: {checkpoint_dir}")
    logging.info(f"Logging to: {checkpoint_dir / 'training.log'}")

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
    train_loader, val_loader, test_loader = load_lattice_dataloaders(
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

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    # Training loop
    agent.train()
    for epoch in tqdm(range(args.epochs), dynamic_ncols=True):
        for step, batch in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            agent.train_step(batch, device)

            # Evaluation
            if (step + 1) % args.eval_interval == 0:
                val_metrics = agent.evaluate(val_loader, device)
                logging.info(
                    f"Epoch {epoch}, Step {step}, Val Success: {val_metrics['success_rate']:.2f}, Avg Shortness: {val_metrics['avg_shortness']}, Avg Time: {val_metrics['avg_time']}, Avg Reward: {val_metrics['avg_reward']}, Avg Steps: {val_metrics['avg_steps']}")

                filename = f"epoch_{epoch}-step_{step}-valSuccess{val_metrics['success_rate']:.2f}.pth"
                torch.save(agent.state_dict(), checkpoint_dir / filename)

                agent.train()

    test_metrics = agent.evaluate(test_loader, device)
    logging.info(
        f"Test Success: {test_metrics['success_rate']:.2f}, Avg Shortness: {test_metrics['avg_shortness']}, Avg Time: {test_metrics['avg_time']}, Avg Reward: {test_metrics['avg_reward']}, Avg Steps: {test_metrics['avg_steps']}")


if __name__ == "__main__":
    main()
