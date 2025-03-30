import argparse
from collections import defaultdict
import logging
from pathlib import Path
import random
from typing import Dict
import yaml

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from load_dataset import load_lattice_dataloader


def evaluate(agent: Agent, val_dataloader, checkpoint_episode: int) -> Dict[str, float]:
    aggregated_metrics = defaultdict(list)

    with torch.no_grad():
        for episode_index, batch in enumerate(tqdm(val_dataloader,
                                        dynamic_ncols=True,
                                        desc=f"Validating Checkpoint {checkpoint_episode}")):
            batch_metrics = agent.evaluate(batch)

            per_batch_log = {
                f"val/{metric}_{checkpoint_episode}": value
                for metric, value in batch_metrics.items()
            }
            per_batch_log["episode"] = episode_index
            wandb.log(per_batch_log)

            for metric, value in batch_metrics.items():
                aggregated_metrics[metric].append(value)

        avg_metrics = {
            f"avg_{metric}": sum(values) / len(values)
            for metric, values in aggregated_metrics.items()
        }

        final_log = {f"val/{metric}": value for metric, value in avg_metrics.items()}
        final_log["checkpoint_episode"] = episode_index
        wandb.log(final_log)

        avg_metrics["checkpoint_episode"] = checkpoint_episode

        return avg_metrics

def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--run-dir", type=str, required=True)

    dist_group = parser.add_mutually_exclusive_group(required=True)
    dist_group.add_argument("--uniform", action="store_true", help="Use a uniform distribution.")
    dist_group.add_argument("--qary", action="store_true", help="Use a q-ary distribution.")
    dist_group.add_argument("--ntrulike", action="store_true", help="Use an NTRU-like distribution.")

    args = parser.parse_args()

    if args.uniform:
        args.dist = "uniform"
    elif args.qary:
        args.dist = "qary"
    elif args.ntrulike:
        args.dist = "ntrulike"

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

    logging.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('We will use the GPU: ' +
                     str(torch.cuda.get_device_name(0)))
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    run_id = Path(args.run_dir).name.replace(":", "_") + f"_test_dim_{args.dim}"
    wandb.init(project="bkz-rl-evaluation", name=run_id, id=run_id, resume="allow")

    data_dir = Path("random_bases")

    # Create DataLoaders
    val_loader = load_lattice_dataloader(
        data_dir=data_dir,
        dimension=args.dim,
        distribution_type=args.dist,
        batch_size=1,
        shuffle=True,
        device=device
    )

    run_dir = Path(args.run_dir)
    checkpoint_files = []
    pretrained_file = None

    # Collect and sort checkpoint files
    for pth_file in run_dir.glob('*.pth'):
        if pth_file.name == 'pretrained.pth':
            pretrained_file = pth_file
        else:
            stem = pth_file.stem
            if stem.startswith('episodes_'):
                try:
                    episode = int(stem.split('_')[1])
                    checkpoint_files.append((episode, pth_file))
                except (IndexError, ValueError):
                    logging.warning(f"Skipping invalid file: {pth_file}")
    checkpoint_files.sort()
    if pretrained_file:
        checkpoint_files = [(0, pretrained_file)] + checkpoint_files

    # Process each checkpoint in order
    for checkpoint_episode, pth_file in checkpoint_files:
        yaml_file = pth_file.with_suffix('.yaml')
        if yaml_file.exists():
            logging.info(f"Skipping {pth_file} as {yaml_file} exists.")
            continue

        logging.info(f"Evaluating {pth_file}...")

        checkpoint = torch.load(pth_file, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        agent_config = checkpoint['agent_config']
        agent_config.batch_size = 1
        agent_config.device = device
        logging.info(agent_config)

        agent = Agent(agent_config=agent_config).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()

        total_params = sum(p.numel() for p in agent.parameters())
        logging.info(f"Total parameters: {total_params}")

        metrics = evaluate(agent, val_loader, checkpoint_episode)
        logging.info(f"Validation metrics:")
        logging.info(str(metrics))

        with open(yaml_file, "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        logging.info(f"Saved metrics to {yaml_file}")


if __name__ == "__main__":
    main()
