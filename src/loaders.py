from torch.utils.data import DataLoader, random_split, Subset
import torch

from src.dataset import ASLDataset
from src.transforms import get_transform


def get_loaders(root_dir, batch_Size, val_split=0.2):
    transforms = get_transform()

    train_dataset_full = ASLDataset(root_dir=root_dir, transform=transforms["train"])
    val_dataset_full = ASLDataset(root_dir=root_dir, transform=transforms["val"])

    dataset_size = len(train_dataset_full)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_idxs, val_idxs = random_split(
        range(dataset_size), [train_size, val_size], generator=generator
    )

    train_subset = Subset(train_dataset_full, train_idxs)
    val_subset = Subset(val_dataset_full, val_idxs)

    train_loader = DataLoader(train_subset, batch_size=batch_Size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_Size, shuffle=False)

    return train_loader, val_loader, train_dataset_full.classes
