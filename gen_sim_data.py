import argparse
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from load_dataset import load_lattice_dataloaders
from ppo_agent import ActorCritic
from reduction_env import ReductionEnvConfig, ReductionEnvironment

def main():
    distributions = ["uniform", "qary", "ntrulike"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-d", "--dim", type=int, default=32)
    parser.add_argument("--distribution", type=str,
                        choices=distributions)
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-block-size", type=int)
    parser.add_argument("--time-limit", type=float, default=10.0)
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

    data_dir = Path("random_bases")
    train_loader, _, _ = load_lattice_dataloaders(
        data_dir=data_dir,
        dimension=args.dim,
        distribution_type=args.distribution,
        batch_size=1,
        shuffle=True,
        device=torch.device("cpu")
    )

    env_config = ReductionEnvConfig(
        basis_dim=args.dim,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        batch_size=1,
        time_limit=args.time_limit
    )

    env = ReductionEnvironment(env_config)

    samples = []

    for _ in tqdm(range(10)):
        for batch in tqdm(train_loader, dynamic_ncols=True):
            state, info = env.reset(options=batch[0])
            
            done = False

            prev_block = 2
            while not done and env_config.max_block_size > prev_block:
                block_size = random.randint(prev_block + 1, env_config.max_block_size)

                action = env._block_to_action(block_size)
                next_state, _, terminated, truncated, next_info = env.step(action)
                
                state_info = ActorCritic.preprocess_inputs(state)
                next_state_info = ActorCritic.preprocess_inputs(next_state)
                samples.append({
                    "previous_gs_norms": state_info["gs_norms"],
                    "next_gs_norms": next_state_info["gs_norms"],
                    "previous_block_size": env._action_to_block(state_info["last_action"]),
                    "next_block_size": env._action_to_block(next_state_info["last_action"]),
                    "time_taken": next_info["time"] - info["time"]
                })
                state = next_state
                info = next_info
                prev_block = block_size

                done = terminated or truncated

    np.save(f"sim_training_data_{args.distribution}_{args.dim}.npy", samples)


if __name__ == "__main__":
    main()
