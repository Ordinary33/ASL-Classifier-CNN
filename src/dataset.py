from torch.utils.data import Dataset
from pathlib import Path
from src import logger_config
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF_PATH = PROJECT_ROOT / "data"

logger = logger_config.setup_logger(name="dataset_logger")


class ASLDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        logger.info(
            f"Dataset initialized with {len(self.dataset)} samples from {root_dir}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    @property
    def classes(self):
        return self.dataset.classes
