import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List

from fpylll import FPLLL
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from agent import Agent
from load_dataset import load_lattice_dataloader


from collections import defaultdict
import copy


def transpose_list_of_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                result[k].append(v)
            else:
                result[k].append(copy.deepcopy(v))

    for k in result:
        if isinstance(result[k][0], dict):
            result[k] = transpose_list_of_dicts(result[k])
        elif all(isinstance(i, list) for i in result[k]):
            result[k] = [item for sublist in result[k] for item in sublist]
    return dict(result)


def recursive_mean(d: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = recursive_mean(v)
        elif isinstance(v, list):
            result[k] = float(np.nanmean((v)))
        else:
            raise ValueError(f"Expected list or dict at key '{k}', got {type(v)}")
    return result


def evaluate(
    agent: Agent,
    dataloader: DataLoader,
    checkpoint_episode: int,
) -> Dict[str, Any]:
    aggregated_metrics = []
    aggregated_raw_logs = []

    with torch.no_grad():
        for index, batch in enumerate(
            tqdm(
                dataloader,
                dynamic_ncols=True,
                desc=f"Validating Checkpoint {checkpoint_episode}",
            )
        ):
            batch_metrics, episode_logs = agent.evaluate(batch)
            aggregated_metrics.append(batch_metrics)
            aggregated_raw_logs.append(episode_logs)

            wandb.log({**batch_metrics, "index": index})
            wandb.log({"raw_val_logs": episode_logs})

        aggregated_metrics = transpose_list_of_dicts(aggregated_metrics)
        avg_metrics = recursive_mean(aggregated_metrics)
        avg_metrics["checkpoint_episode"] = checkpoint_episode
        wandb.log(avg_metrics)

        return avg_metrics, aggregated_raw_logs


def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--run-dir", type=str, required=True)

    dist_group = parser.add_mutually_exclusive_group(required=True)
    dist_group.add_argument(
        "--uniform",
        action="store_const",
        const="uniform",
        dest="dist",
        help="Use a uniform distribution.",
    )
    dist_group.add_argument(
        "--qary",
        action="store_const",
        const="qary",
        dest="dist",
        help="Use a q-ary distribution.",
    )
    dist_group.add_argument(
        "--ntrulike",
        action="store_const",
        const="ntrulike",
        dest="dist",
        help="Use an NTRU-like distribution.",
    )
    dist_group.add_argument(
        "--knapsack",
        action="store_const",
        const="knapsack",
        dest="dist",
        help="Use a knapsack distribution.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
        logging.info("We will use the GPU: " + str(torch.cuda.get_device_name(0)))
    else:
        logging.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    run_id = f"{Path(args.run_dir).name}/dim_{args.dim}-dist_{args.dist}"
    wandb.init(project="bkz-rl-evaluation", name=run_id)

    data_dir = Path("random_bases")

    # Create DataLoaders
    val_loader = load_lattice_dataloader(
        data_dir=data_dir,
        dimension=args.dim,
        distribution_type=args.dist,
        batch_size=1,
        shuffle=False,
        device=device,
    )

    run_dir = Path(args.run_dir)

    # Create reports directory if it doesn't exist
    reports_dir = run_dir / "reports" / f"dim_{args.dim}-dist_{args.dist}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = []

    # Collect and sort checkpoint files
    for pth_file in run_dir.glob("episodes_*.pth"):
        try:
            episode = int(pth_file.stem.split("_")[1])
            checkpoint_files.append((episode, pth_file))
        except (IndexError, ValueError):
            logging.warning(f"Skipping invalid file: {pth_file}")

    checkpoint_files.sort()

    # Process each checkpoint in order
    for checkpoint_episode, pth_file in checkpoint_files:
        json_filename = reports_dir / f"episode_{checkpoint_episode}.json"
        if json_filename.exists():
            logging.info(f"Skipping {pth_file} as {json_filename} exists.")
            continue

        logging.info(f"Evaluating {pth_file}...")

        checkpoint = torch.load(pth_file, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        agent_config = checkpoint["agent_config"]
        agent_config.batch_size = 1
        agent_config.device = device
        logging.info(agent_config)

        agent = Agent(agent_config=agent_config).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()

        total_params = sum(p.numel() for p in agent.parameters())
        logging.info(f"Total parameters: {total_params}")

        log_data = {}
        log_data["agent_config"] = str(agent_config)

        avg_metrics, aggregated_raw_logs = evaluate(agent, val_loader, checkpoint_episode)
        logging.info(f"Validation metrics:")
        logging.info(str(avg_metrics))
        log_data["avg_metrics"] = avg_metrics
        log_data["raw_logs"] = aggregated_raw_logs

        with open(json_filename, "w") as json_file:
            json.dump(log_data, json_file, indent=4, sort_keys=False)

        logging.info(f"Saved logs to {json_filename}")
        wandb.save(json_filename)


if __name__ == "__main__":
    main()
