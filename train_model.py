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

from agent import Agent, AgentConfig, PPOConfig, SimulatorConfig
from reduction_env import ReductionEnvConfig


def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser(
        description="Reinforcement Learning for BKZ Lattice Reduction."
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes. Set to a negative value for infinite training",
    )
    parser.add_argument(
        "--chkpt-interval", type=int, default=1000, help="Checkpoint saving interval."
    )

    env_args = parser.add_argument_group("Training Environment Settings")
    env_args.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Time limit before environment truncates run.",
    )
    env_args.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training."
    )
    env_args.add_argument(
        "--train-min-dim",
        type=int,
        required=True,
        help="Minimum basis dimension for training instances.",
    )
    env_args.add_argument(
        "--train-max-dim",
        type=int,
        required=True,
        help="Maximum basis dimension for training instances.",
    )
    env_args.add_argument(
        "--max-block-size", type=int, help="Maximum block size for reduction."
    )

    dist_args = env_args.add_mutually_exclusive_group(required=True)
    dist_args.add_argument(
        "--uniform", action="store_true", help="Use a uniform distribution."
    )
    dist_args.add_argument(
        "--qary", action="store_true", help="Use a q-ary distribution."
    )
    dist_args.add_argument(
        "--ntrulike", action="store_true", help="Use an NTRU-like distribution."
    )

    arch_args = parser.add_argument_group("Architecture Settings")
    arch_args.add_argument(
        "--simulator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a simulator for training.",
    )
    arch_args.add_argument(
        "--net-dim",
        type=int,
        required=True,
        help="Maximum input dimension for neural network architecture.",
    )

    pred_args = arch_args.add_mutually_exclusive_group(required=True)
    pred_args.add_argument(
        "--continuous", action="store_true", help="Use continuous prediction type."
    )
    pred_args.add_argument(
        "--discrete", action="store_true", help="Use discrete prediction type."
    )

    reward_args = parser.add_argument_group("Reward Weights")
    reward_args.add_argument(
        "--time-penalty-weight",
        type=float,
        default=-1.0,
        help="Weight for time penalty in the reward function.",
    )
    reward_args.add_argument(
        "--defect-reward-weight",
        type=float,
        default=0.1,
        help="Weight for (log) orthogonality defect reduction in the reward function.",
    )
    reward_args.add_argument(
        "--length-reward-weight",
        type=float,
        default=1.0,
        help="Weight for shortest vector length reduction in the reward function.",
    )

    simulator_args = parser.add_argument_group(
        "Simulator Weights (only used if --simulator is enabled)"
    )
    simulator_args.add_argument(
        "--simulator-lr",
        type=float,
        default=1e-5,
        help="Learning rate for simulator optimizer.",
    )
    simulator_args.add_argument(
        "--simulator-reward-weight",
        type=float,
        default=0.1,
        help="Weight for simulator reward term (only used if --simulator is enabled).",
    )
    simulator_args.add_argument(
        "--simulator-basis-weight",
        type=float,
        default=1.0,
        help="Weight for simulator basis loss term (only used if --simulator is enabled).",
    )
    simulator_args.add_argument(
        "--simulator-time-weight",
        type=float,
        default=1.0,
        help="Weight for simulator time loss term (only used if --simulator is enabled).",
    )
    simulator_args.add_argument(
        "--simulator-inverse-weight",
        type=float,
        default=0.05,
        help="Weight for simulator inverse loss term (only used if --simulator is enabled). The inverse norm will inherently be very noisy since there may not be a unique action that maps to the same state.",
    )

    ppo_args = parser.add_argument_group("PPO Training Parameters")
    ppo_args.add_argument(
        "--minibatch",
        type=int,
        default=64,
        help="Minibatch size for updating weights in PPO.",
    )
    ppo_args.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for PPO."
    )
    ppo_args.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Number of epochs for each update in PPO.",
    )
    ppo_args.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="Clipping epsilon for PPO."
    )
    ppo_args.add_argument(
        "--clip-grad-norm", type=float, default=0.5, help="Clipping norm for PPO."
    )
    ppo_args.add_argument(
        "--ppo-gamma", type=float, default=0.99, help="Discount factor for PPO."
    )
    ppo_args.add_argument(
        "--ppo-gae-lambda", type=float, default=0.95, help="Lambda parameter for GAE."
    )

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

    # if not args.simulator:
    #     args.simulator_reward_weight = 0.0

    # reward_weight_sum = (
    #     abs(args.time_penalty_weight)
    #     + abs(args.defect_reward_weight)
    #     + abs(args.length_reward_weight)
    #     + abs(args.simulator_reward_weight)
    # )
    # args.time_penalty_weight /= reward_weight_sum
    # args.defect_reward_weight /= reward_weight_sum
    # args.length_reward_weight /= reward_weight_sum
    # args.simulator_reward_weight /= reward_weight_sum

    # simulator_weight_sum = (
    #     abs(args.simulator_reward_weight)
    #     + abs(args.simulator_gs_norm_weight)
    #     + abs(args.simulator_time_weight)
    #     + abs(args.simulator_inverse_weight)
    # )
    # args.simulator_gs_norm_weight /= simulator_weight_sum
    # args.simulator_time_weight /= simulator_weight_sum
    # args.simulator_inverse_weight /= simulator_weight_sum

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

    run_name = f"dim-{args.net_dim}_{args.train_min_dim}_{args.train_max_dim}_{start_timestamp}"
    wandb.init(project="bkz-rl-training", name=run_name)

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

    ppo_config = PPOConfig(
        minibatch_size=args.minibatch,
        lr=args.learning_rate,
        epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
    )
    simulator_config = SimulatorConfig(
        lr=args.simulator_lr,
        basis_weight=args.simulator_basis_weight,
        time_weight=args.simulator_time_weight,
        inverse_weight=args.simulator_inverse_weight,
    )
    agent_config = AgentConfig(
        ppo_config=ppo_config,
        device=device,
        batch_size=args.batch_size,
        env_config=env_config,
        simulator=args.simulator,
        pred_type=args.pred_type,
        simulator_reward_weight=args.simulator_reward_weight,
        simulator_config=simulator_config,
    )
    agent = Agent(agent_config=agent_config).to(device)

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    agent.save(checkpoint_dir / "episodes_0.pth")
    logging.info("Saved pretrained model as episodes_0.pth")

    agent.train()

    episode = 0
    progress_bar = tqdm(
        desc="Training episodes",
        dynamic_ncols=True,
        initial=0,
        total=args.episodes if args.episodes >= 0 else None
    )
    while args.episodes < 0 or episode < args.episodes:
        episode_metrics = agent.collect_experiences()
        update_metrics = agent.update()

        for metric in episode_metrics:
            wandb.log(metric)

        wandb.log(update_metrics)

        episode += 1
        progress_bar.update(1)
        progress_bar.set_description(f"Training episode: {episode}")

        if episode % args.chkpt_interval == 0 and episode != args.episodes:
            agent.save(checkpoint_dir / f"episodes_{episode}.pth")
            logging.info(f"Saved checkpoint at episode {episode}")

    progress_bar.close()

    logging.info(f"Training ended at episode {episode}")
    agent.save(checkpoint_dir / f"episodes_{episode}.pth")
    logging.info(f"Saved final model at episode {episode}")


if __name__ == "__main__":
    main()
