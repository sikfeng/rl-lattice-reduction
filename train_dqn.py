import datetime
import logging
from pathlib import Path
import random

from fpylll import FPLLL
import numpy as np
import torch
from tqdm import tqdm

from reduction_env import BKZEnvConfig
from agents.dqn_agent import DQNAgent, DQNConfig
from load_dataset import load_lattice_dataloaders


def main():
    FPLLL.set_precision(1000)

    seed_val = 0
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed_all(seed_val)

    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_dir = Path(f"checkpoint/ppo-model-{start_timestamp}")
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
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('We will use the GPU: ' + str(torch.cuda.get_device_name(0)))
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Define dataset parameters
    dimension = 10  # Must match environment's basis_dim
    data_dir = Path("random_bases")
    distribution_type = "uniform"

    # Create DataLoaders
    train_loader, val_loader, test_loader = load_lattice_dataloaders(
        data_dir=data_dir,
        dimension=dimension,
        distribution_type=distribution_type,
        batch_size=1,
        shuffle=True,
        device=device
    )

    # Environment and agent configuration
    env_config = BKZEnvConfig(basis_dim=dimension, min_block_size=2, max_block_size=2)
    dqn_config = DQNConfig(env_config=env_config)
    agent = DQNAgent(dqn_config=dqn_config)
    agent.train()

    total_params = sum(p.numel() for p in agent.parameters())
    logging.info(f"Total parameters: {total_params}")

    # Training loop
    epochs = 2
    best_success = 0.0

    for epoch in tqdm(range(epochs)):
        for step, batch in enumerate(tqdm(train_loader)):
            agent.train_step(batch)
            
            # Evaluation
            if (step + 1) % 1000 == 0:
                val_metrics = agent.evaluate(val_loader,device)
                test_metrics = agent.evaluate(test_loader, device)
                logging.info(f"Epoch {epoch}, Step {step}, Val Success: {val_metrics['success_rate']:.2f}, Test Success: {test_metrics['success_rate']:.2f}")
                
                # Save model at every evaluation with details in the filename
                filename = f"epoch_{epoch}-step_{step}-valSuccess{val_metrics['success_rate']:.2f}.pth"
                torch.save(agent.state_dict(), checkpoint_dir / filename)
                
                # Additionally, save the best model if the current success is higher than before
                if val_metrics['success_rate'] > best_success:
                    best_success = val_metrics['success_rate']
                    best_filename = "best_agent.pth"
                    torch.save(agent.state_dict(), checkpoint_dir / best_filename)

                agent.train()
    
    if __name__ == "__main__":
        main()