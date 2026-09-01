import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Tuple
from data_layer.dataset_manager import DatasetManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SatQueryDataset(Dataset):
    """
    PyTorch Dataset for SatQuery Remote Sensing Tasks (VQA, Grounding, Change Detection, SAR Fusion).
    """

    def __init__(self, dataset_name: str, split: str = 'train', data_path: str = './data'):
        self.dataset_manager = DatasetManager(data_path=data_path)
        self.data_dict = self.dataset_manager.load_dataset(dataset_name, split=split)
        self.items = self.data_dict.get('items', [])
        self.dataset_name = dataset_name
        self.split = split
        logger.info(f"SatQueryDataset initialized for {dataset_name} ({split} split, {len(self.items)} samples)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        item = self.items[idx]
        
        # Generate synthetic 3x64x64 tensor representing satellite imagery
        image_np = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)

        # Prepare target label dict depending on item task type
        target = {
            'id': item.get('id', f'sample_{idx}'),
            'query': item.get('question', item.get('query', 'Analyze image')),
            'task_type': item.get('task_type', 'vqa')
        }

        if 'answer' in item:
            target['answer'] = item['answer']
        if 'bbox' in item:
            target['bbox'] = torch.tensor(item['bbox'], dtype=torch.float32)
        if 'change_percentage' in item:
            target['change_percentage'] = torch.tensor(item['change_percentage'], dtype=torch.float32)
        if 'sar_vv_db' in item:
            target['sar_features'] = torch.tensor([item['sar_vv_db'], item['sar_vh_db']], dtype=torch.float32)

        return image_tensor, target


def create_dataloader(
    dataset_name: str,
    split: str = 'train',
    batch_size: int = 4,
    shuffle: bool = True,
    data_path: str = './data'
) -> DataLoader:
    """
    Factory helper function to instantiate PyTorch DataLoaders.
    """
    ds = SatQueryDataset(dataset_name=dataset_name, split=split, data_path=data_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_custom_collate_fn
    )


def _custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
