import argparse
from collections import defaultdict
import logging
from pathlib import Path
import random
import yaml

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm
import wandb

from ppo_agent import PPOAgent
from load_dataset import load_lattice_dataloaders


def evaluate(agent: PPOAgent, val_dataloader, test_dataloader, checkpoint_episode: int) -> dict:
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)

    with torch.no_grad():
        # Validation evaluation
        for ep, batch in enumerate(tqdm(val_dataloader, dynamic_ncols=True)):
            batch_metrics = agent.evaluate(batch)
            # Log per-batch metrics
            logged_metrics = {f"val/{k}_{checkpoint_episode}": v for k, v in batch_metrics.items()}
            logged_metrics[f"val/episode"] = ep
            wandb.log(logged_metrics)
            # Accumulate for aggregation
            for k, v in batch_metrics.items():
                val_metrics[k].append(v)
        
        # Aggregate validation metrics
        aggregated_val = {}
        for k in val_metrics:
            avg = sum(val_metrics[k]) / len(val_metrics[k])
            aggregated_val[f'avg_{k}'] = avg

        # Test evaluation
        for ep, batch in enumerate(tqdm(test_dataloader, dynamic_ncols=True)):
            batch_metrics = agent.evaluate(batch)
            # Log per-batch metrics
            logged_metrics = {f"test/{k}_{checkpoint_episode}": v for k, v in batch_metrics.items()}
            logged_metrics[f"test/episode"] = ep
            wandb.log(logged_metrics)
            # Accumulate for aggregation
            for k, v in batch_metrics.items():
                test_metrics[k].append(v)
        
        # Aggregate test metrics
        aggregated_test = {}
        for k in test_metrics:
            avg = sum(test_metrics[k]) / len(test_metrics[k])
            aggregated_test[f'avg_{k}'] = avg

    return {
        'val': aggregated_val,
        'test': aggregated_test
    }

def main():
    FPLLL.set_precision(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dist", type=str, required=True, choices=["uniform", "qary", "ntrulike"])
    args = parser.parse_args()

    logging.info(args)

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

    run_id = Path(args.run_dir).name.replace(":", "_")
    wandb.init(project="bkz-rl-evaluation", name=run_id, id=run_id, resume="allow")

    data_dir = Path("random_bases")

    # Create DataLoaders
    _, val_loader, test_loader = load_lattice_dataloaders(
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
        ppo_config = checkpoint['ppo_config']
        agent = PPOAgent(ppo_config=ppo_config, device=device).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()

        total_params = sum(p.numel() for p in agent.parameters())
        logging.info(f"Total parameters: {total_params}")

        metrics = evaluate(agent, val_loader, test_loader, checkpoint_episode)
        logging.info(f"Validation metrics:")
        logging.info(str(metrics["val"]))
        logging.info(f"Test metrics:")
        logging.info(str(metrics["test"]))

        with open(yaml_file, "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        logging.info(f"Saved metrics to {yaml_file}")


if __name__ == "__main__":
    main()
