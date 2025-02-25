from pathlib import Path

import numpy as np
from tensordict import TensorDict
import torch
from torch.utils.data import Dataset, DataLoader

class LatticeBaseDataset(Dataset):
    """Dataset for lattice bases with precomputed shortest vectors and LLL reduction"""
    
    def __init__(self, file_path, transform=None, device=None):
        """
        Initialize the dataset from a pickle file.
        
        Args:
            file_path (str): Path to the save file containing the lattice data
            transform (callable, optional): Optional transform to apply to the data
            device (torch.device, optional): Device to load tensors to (default: None = CPU)
        """
        self.data = np.load(file_path, allow_pickle=True)
        
        self.transform = transform
        self.device = device or torch.device('cpu')
        
        # Extract dimension from the first sample
        self.dim = self.data[0]['basis'].shape[0]
        
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
        tensor_sample = TensorDict({
            'basis': torch.tensor(sample['basis'], dtype=torch.float32),
            'shortest_vector': torch.tensor(sample['shortest_vector'], dtype=torch.float32),
            'lll_reduced_basis': torch.tensor(sample['lll_reduced_basis'], dtype=torch.float32),
            'lll_log_defect': torch.tensor(sample['lll_log_defect'], dtype=torch.float32)
        }, batch_size=[])
        
        # Apply transforms if specified
        if self.transform:
            tensor_sample = self.transform(tensor_sample)
            
        return tensor_sample


def load_lattice_dataloader(data_dir, dimension, distribution_type, 
                           split='train', batch_size=32, shuffle=True, 
                           num_workers=0, transform=None, device=None):
    """
    Create a DataLoader for lattice basis datasets using pathlib for path handling.
    
    Args:
        data_dir (str or Path): Directory containing the dataset files
        dimension (int): Dimension of the lattice bases
        distribution_type (str): Type of distribution used to generate the bases
                                (e.g., 'uniform', 'exponential', 'convex', 'ajtai')
        split (str): Dataset split to load ('train', 'val', or 'test')
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes for data loading
        transform (callable, optional): Transform to apply to the data
        device (torch.device, optional): Device to load the tensors to
        
    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset
    """
    file_path = Path(data_dir) / f"{split}_dim_{dimension}_type_{distribution_type}.npy"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    dataset = LatticeBaseDataset(file_path, transform=transform, device=device)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == 'cuda'),
        collate_fn=TensorDict.stack  # Stack TensorDicts into batches
    )

def load_lattice_dataloaders(data_dir, dimension, distribution_type, 
                             batch_size=32, shuffle=True, num_workers=0, 
                             transform=None, device=None):
    train_dataloader = load_lattice_dataloader(data_dir=data_dir, dimension=dimension, 
                                               distribution_type=distribution_type, 
                                               split="train", batch_size=batch_size, 
                                               shuffle=shuffle, num_workers=num_workers, 
                                               transform=transform, device=device)
    val_dataloader = load_lattice_dataloader(data_dir=data_dir, dimension=dimension, 
                                               distribution_type=distribution_type, 
                                               split="val", batch_size=batch_size, 
                                               shuffle=shuffle, num_workers=num_workers, 
                                               transform=transform, device=device)
    test_dataloader = load_lattice_dataloader(data_dir=data_dir, dimension=dimension, 
                                               distribution_type=distribution_type, 
                                               split="test", batch_size=batch_size, 
                                               shuffle=shuffle, num_workers=num_workers, 
                                               transform=transform, device=device)
    
    return train_dataloader, val_dataloader, test_dataloader