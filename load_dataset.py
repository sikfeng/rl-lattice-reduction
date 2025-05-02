from pathlib import Path

import numpy as np
from tensordict import TensorDict
import torch
from torch.utils.data import Dataset, DataLoader


class LatticeBaseDataset(Dataset):
    """Dataset for lattice bases with precomputed shortest vectors and LLL reduction"""

    def __init__(self, file_path, dimension, transform=None, device=None):
        """
        Initialize the dataset from a pickle file.

        Args:
            file_path (str): Path to the save file containing the lattice data
            transform (callable, optional): Optional transform to apply to the data
            device (torch.device, optional): Device to load tensors to (default: None = CPU)
        """
        self.data = np.load(file_path, allow_pickle=True)

        self.transform = transform
        self.device = device or torch.device("cpu")

        self.dim = dimension

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            dict: Dictionary containing tensors of the basis, shortest vector,
                 LLL-reduced basis, and orthogonality defect
        """
        sample = self.data[idx]

        # Convert numpy arrays to PyTorch tensors and move to device
        tensor_sample = TensorDict(
            {key: torch.tensor(sample[key], dtype=torch.float32) for key in sample},
            batch_size=[],
        )

        # Apply transforms if specified
        if self.transform:
            tensor_sample = self.transform(tensor_sample)

        return tensor_sample


def load_lattice_dataloader(
    data_dir,
    dimension,
    distribution_type,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    transform=None,
    device=None,
):
    file_path = Path(data_dir) / f"dim_{dimension}_type_{distribution_type}.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    dataset = LatticeBaseDataset(
        file_path, dimension, transform=transform, device=device
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        collate_fn=TensorDict.stack,  # Stack TensorDicts into batches
    )
