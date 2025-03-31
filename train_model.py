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

from agent import Agent, AgentConfig, PPOConfig
from reduction_env import ReductionEnvConfig


def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser(description="Reinforcement Learning for BKZ Lattice Reduction.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--chkpt-interval", type=int, default=1000, help="Checkpoint saving interval.")
    parser.add_argument("--max-block-size", type=int, help="Maximum block size for reduction.")
    parser.add_argument("--net-dim", type=int, required=True, help="Maximum input dimension for neural network architecture.")
    parser.add_argument("--train-min-dim", type=int, required=True, help="Minimum basis dimension for training instances.")
    parser.add_argument("--train-max-dim", type=int, required=True, help="Maximum basis dimension for training instances.")
    parser.add_argument("--time-penalty-weight", type=float, default=-1.0, help="Weight for time penalty in the reward function.")
    parser.add_argument("--defect-reward-weight", type=float, default=0.1, help="Weight for (log) orthogonality defect reduction in the reward function.")
    parser.add_argument("--length-reward-weight", type=float, default=1.0, help="Weight for shortest vector length reduction (of the resulting basis) in the reward function.")
    parser.add_argument("--time-limit", type=float, default=300.0, help="Time limit before environment truncates run.")
    parser.add_argument("--simulator", action=argparse.BooleanOptionalAction, default=False, help="Use a simulator for training.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--minibatch", type=int, default=1, help="Minibatch size for updating weights in PPO.")

    dist_group = parser.add_mutually_exclusive_group(required=True)
    dist_group.add_argument("--uniform", action="store_true", help="Use a uniform distribution.")
    dist_group.add_argument("--qary", action="store_true", help="Use a q-ary distribution.")
    dist_group.add_argument("--ntrulike", action="store_true", help="Use an NTRU-like distribution.")

    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument("--continuous", action="store_true", help="Use continuous prediction type.")
    pred_group.add_argument("--discrete", action="store_true", help="Use discrete prediction type.")

    args = parser.parse_args()

    if args.train_min_dim > args.train_max_dim:
        raise ValueError("train_min_dim must be <= train_max_dim")
    if args.train_max_dim > args.net_dim:
        raise ValueError("train_max_dim cannot exceed net_dim")
    if args.max_block_size is None:
        args.max_block_size = args.train_max_dim  # Set to max training dimension
    if args.max_block_size > args.train_max_dim:
        raise ValueError("max_block_size must be <= train_max_dim")
    if args.max_block_size > args.net_dim:
        raise ValueError("max_block_size must be at most dim.")
    if 2 > args.max_block_size:
        raise ValueError("max_block_size cannot be less than 2.")

    if args.uniform:
        args.dist = "uniform"
    elif args.qary:
        args.dist = "qary"
    elif args.ntrulike:
        args.dist = "ntrulike"

    # Determine selected prediction type
    if args.continuous:
        args.pred_type = "continuous"
    else:
        args.pred_type = "discrete"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_dir = Path(
        f"checkpoint/dim-{args.net_dim}_{args.train_min_dim}_{args.train_max_dim}_{start_timestamp}")
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

    wandb.init(project="bkz-rl-training", name=f"dim-{args.net_dim}_{args.train_min_dim}_{args.train_max_dim}_{start_timestamp}")

    env_config = ReductionEnvConfig(
        net_dim=args.net_dim,
        train_min_dim=args.train_min_dim,
        train_max_dim=args.train_max_dim,
        max_block_size=args.max_block_size,
        time_penalty_weight=args.time_penalty_weight,
        defect_reward_weight=args.defect_reward_weight,
        length_reward_weight=args.length_reward_weight,
        time_limit=args.time_limit,
        distribution=args.dist,
        batch_size=args.batch_size,
    )

    ppo_config = PPOConfig(minibatch_size=args.minibatch)
    agent_config = AgentConfig(
        ppo_config=ppo_config,
        device=device,
        batch_size=args.batch_size,
        env_config=env_config,
        simulator=args.simulator,
        pred_type=args.pred_type
    )
    agent = Agent(agent_config=agent_config).to(device)

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    agent.save(checkpoint_dir / "pretrained.pth")

    agent.train()
    for episode in (tqdm(range(args.episodes), dynamic_ncols=True)):
        episode_metrics = agent.collect_experiences()
        update_metrics = agent.update()

        for metric in episode_metrics:
            wandb.log(metric)

        wandb.log(update_metrics)

        if (episode + 1) % args.chkpt_interval == 0:
            agent.save(checkpoint_dir / f"episodes_{episode}.pth")


if __name__ == "__main__":
    main()
