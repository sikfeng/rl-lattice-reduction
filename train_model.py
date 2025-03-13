import argparse
import datetime
import logging
from pathlib import Path
import random

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent, PPOConfig
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
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-block-size", type=int)
    parser.add_argument("--model", type=str, choices=["ppo"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-penalty-weight", type=float, default=1.0)
    parser.add_argument("--defect-reward-weight", type=float, default=0.1)
    parser.add_argument("--length-reward-weight", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=1.0)
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
        raise ValueError(
            "min_block_size cannot be greater than max_block_size.")

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.model == "ppo":
        checkpoint_dir = Path(
            f"checkpoint/ppo-model_dim-{args.dim}_{start_timestamp}")
    else:
        raise ValueError("Invalid model type provided.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(checkpoint_dir / "training.log", mode='w'),
            logging.StreamHandler()
        ]
    )

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
        batch_size=args.batch_size,
        shuffle=True,
        device=device
    )

    # Environment and agent configuration
    env_config = ReductionEnvConfig(
        basis_dim=args.dim,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        batch_size=args.batch_size,
        time_penalty_weight=args.time_penalty_weight,
        defect_reward_weight=args.defect_reward_weight,
        length_reward_weight=args.length_reward_weight,
        time_limit=args.time_limit
    )

    if args.model == "ppo":
        ppo_config = PPOConfig(env_config=env_config)
        agent = PPOAgent(ppo_config=ppo_config).to(device)

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    # Training loop
    val_metrics = agent.evaluate(val_loader, device)
    logging.info(
        f"Pretraining, Val Success: {val_metrics['success_rate']:.2f}, Avg Shortness: {val_metrics['avg_shortness']}, Avg Time: {val_metrics['avg_time']}, Avg Reward: {val_metrics['avg_reward']}, Avg Steps: {val_metrics['avg_steps']}")

    # Save model at every evaluation with details in the filename
    filename = f"pretraining-valSuccess{val_metrics['success_rate']:.2f}.pth"
    torch.save(agent.state_dict(), checkpoint_dir / filename)

    # Additionally, save the best model if the current success is higher than before
    best_success = val_metrics['success_rate']
    best_filename = "best_agent.pth"
    torch.save(agent.state_dict(), checkpoint_dir / best_filename)

    agent.train()

    for epoch in tqdm(range(args.epochs), dynamic_ncols=True):
        for step, batch in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            agent.train_step(batch, device)

            # Evaluation
            if (step + 1) % (args.eval_interval // args.batch_size) == 0:
                val_metrics = agent.evaluate(val_loader, device)
                logging.info(
                    f"Epoch {epoch}, Step {step}, Val Success: {val_metrics['success_rate']:.2f}, Avg Shortness: {val_metrics['avg_shortness']}, Avg Time: {val_metrics['avg_time']}, Avg Reward: {val_metrics['avg_reward']}, Avg Steps: {val_metrics['avg_steps']}")

                filename = f"epoch_{epoch}-step_{step}-valSuccess{val_metrics['success_rate']:.2f}.pth"
                torch.save(agent.state_dict(), checkpoint_dir / filename)

                if val_metrics['success_rate'] > best_success:
                    best_success = val_metrics['success_rate']
                    best_filename = "best_agent.pth"
                    torch.save(agent.state_dict(),
                               checkpoint_dir / best_filename)

                agent.train()

    logging.info(f"Best Val Success: {best_success:.2f}")

    if args.model == "ppo":
        best_agent = PPOAgent(ppo_config=ppo_config).to(device)

    best_agent.load_state_dict(torch.load(checkpoint_dir / best_filename))
    test_metrics = best_agent.evaluate(test_loader, device)
    logging.info(
        f"Test Success: {test_metrics['success_rate']:.2f}, Avg Shortness: {test_metrics['avg_shortness']}, Avg Time: {test_metrics['avg_time']}, Avg Reward: {test_metrics['avg_reward']}, Avg Steps: {test_metrics['avg_steps']}")


if __name__ == "__main__":
    main()
