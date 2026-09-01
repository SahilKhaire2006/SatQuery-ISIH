import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetManager:
    """
    Manages remote sensing datasets, data loading, splits, and synthetic bootstrapping for SatQuery.
    """

    def __init__(self, data_path: str = './data'):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.datasets = {
            'BigEarthNet': {
                'type': 'multi-label classification',
                'modalities': ['Sentinel-1', 'Sentinel-2'],
                'size': '590,326 images'
            },
            'RSVQA': {
                'type': 'VQA',
                'task': 'remote sensing VQA',
                'size': 'evaluation set'
            },
            'CDVQA': {
                'type': 'Change Detection VQA',
                'task': 'temporal analysis',
                'size': 'evaluation set'
            },
            'ISRO_SAC': {
                'type': 'SAR + Optical',
                'task': 'multi-modal fusion',
                'source': 'ISRO SAC eval set'
            },
            'CARTOSAT': {
                'type': 'high-resolution optical',
                'task': 'held-out test labels',
                'source': 'CARTOSAT-2.5'
            }
        }

    def list_datasets(self) -> List[str]:
        """List available datasets"""
        return list(self.datasets.keys())

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a dataset"""
        return self.datasets.get(dataset_name)

    def bootstrap_synthetic_dataset(self, dataset_name: str, num_samples: int = 50) -> Path:
        """
        Generate local synthetic benchmark/training samples for testing local pipelines without manual download.
        """
        target_dir = self.data_path / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.json"

        if manifest_path.exists():
            logger.info(f"Dataset '{dataset_name}' manifest already exists at {manifest_path}")
            return manifest_path

        items = []
        logger.info(f"Bootstrapping synthetic {dataset_name} dataset with {num_samples} samples...")

        for i in range(num_samples):
            sample_id = f"sample_{i:04d}"
            item = {'id': sample_id}

            if dataset_name == 'RSVQA':
                questions = [
                    "What type of land cover is visible?",
                    "How many buildings are in this area?",
                    "Is there a river or water body present?",
                    "Describe the vegetation density."
                ]
                answers = [
                    "Dense urban development",
                    f"{random.randint(2, 12)} structures",
                    "Yes, a water stream is visible.",
                    "High vegetation coverage."
                ]
                idx = random.randint(0, len(questions) - 1)
                item.update({
                    'question': questions[idx],
                    'answer': answers[idx],
                    'task_type': 'vqa'
                })

            elif dataset_name == 'CDVQA':
                item.update({
                    'query': "Detect land surface changes",
                    'change_percentage': round(random.uniform(5.0, 45.0), 2),
                    'comparison_type': 'before_after',
                    'task_type': 'change_detection'
                })

            elif dataset_name in ['BigEarthNet', 'ISRO_SAC']:
                item.update({
                    'query': "Optical-SAR multi-modal land classification",
                    'modalities': ['optical', 'sar'],
                    'sar_vv_db': round(random.uniform(-18.0, -8.0), 2),
                    'sar_vh_db': round(random.uniform(-25.0, -12.0), 2),
                    'task_type': 'sar_fusion'
                })

            else:  # Grounding / CARTOSAT
                item.update({
                    'query': "Locate target structure",
                    'bbox': [10, 10, 80, 80],
                    'label': 'building',
                    'task_type': 'grounding'
                })

            items.append(item)

        manifest = {
            'dataset_name': dataset_name,
            'num_samples': len(items),
            'items': items
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Synthetic dataset manifest created at {manifest_path}")
        return manifest_path

    def load_dataset(self, dataset_name: str, split: str = 'train') -> Dict:
        """
        Load dataset manifest and split
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        dataset_dir = self.data_path / dataset_name
        manifest_path = dataset_dir / "manifest.json"

        if not manifest_path.exists():
            manifest_path = self.bootstrap_synthetic_dataset(dataset_name, num_samples=30)

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        items = manifest.get('items', [])
        train_idx, val_idx, test_idx = self._get_split_indices(len(items))

        if split == 'train':
            selected_items = [items[i] for i in train_idx]
        elif split == 'val':
            selected_items = [items[i] for i in val_idx]
        else:
            selected_items = [items[i] for i in test_idx]

        return {
            'name': dataset_name,
            'split': split,
            'total_items': len(items),
            'items': selected_items,
            'loaded': True
        }

    def _get_split_indices(self, total: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[List[int], List[int], List[int]]:
        """Split indices deterministically into train/val/test"""
        indices = list(range(total))
        train_cutoff = int(total * train_ratio)
        val_cutoff = int(total * (train_ratio + val_ratio))
        return indices[:train_cutoff], indices[train_cutoff:val_cutoff], indices[val_cutoff:]

    def validate_dataset(self, dataset_name: str) -> bool:
        """Validate dataset integrity"""
        dataset_path = self.data_path / dataset_name
        return dataset_path.exists()
