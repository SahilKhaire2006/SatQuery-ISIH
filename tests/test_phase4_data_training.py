import sys
import os
import asyncio
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_layer.dataset_manager import DatasetManager
from training.data_loaders import SatQueryDataset, create_dataloader
from training.train_vqa import run_vqa_training
from training.train_grounding import run_grounding_training
from training.train_change import run_change_training
from training.train_sar_fusion import run_sar_fusion_training
from evaluation.vqa_metrics import compute_vqa_metrics
from evaluation.grounding_metrics import compute_grounding_metrics
from evaluation.change_metrics import compute_change_metrics
from evaluation.benchmark_runner import BenchmarkRunner


def test_dataset_manager_and_splits():
    dm = DatasetManager()
    assert 'RSVQA' in dm.list_datasets()
    data = dm.load_dataset('RSVQA', split='train')
    assert data['loaded'] is True
    assert len(data['items']) > 0


def test_pytorch_dataloader():
    loader = create_dataloader('RSVQA', split='train', batch_size=2)
    for images, targets in loader:
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] <= 2
        assert len(targets) > 0
        break


def test_vqa_training_loop():
    result = run_vqa_training(epochs=1, batch_size=2)
    assert result['epochs'] == 1
    assert 'checkpoint' in result
    assert os.path.exists(result['checkpoint'])


def test_sar_fusion_training_loop():
    result = run_sar_fusion_training(epochs=1, batch_size=2)
    assert result['epochs'] == 1
    assert 'checkpoint' in result
    assert os.path.exists(result['checkpoint'])


def test_evaluation_metrics():
    vqa_res = compute_vqa_metrics(["dense urban"], ["dense urban"])
    assert vqa_res['accuracy'] == 1.0

    g_res = compute_grounding_metrics([[10, 10, 50, 50]], [[10, 10, 50, 50]])
    assert g_res['mean_iou'] > 0.9

    c_res = compute_change_metrics([20.0], [22.0])
    assert c_res['f1_score'] == 1.0


def test_benchmark_runner():
    async def _run():
        runner = BenchmarkRunner()
        report = await runner.run_benchmark('RSVQA', split='test')
        assert report['dataset'] == 'RSVQA'
        assert report['samples_evaluated'] > 0
        assert 'metrics' in report

    asyncio.run(_run())


if __name__ == '__main__':
    test_dataset_manager_and_splits()
    test_pytorch_dataloader()
    test_vqa_training_loop()
    test_sar_fusion_training_loop()
    test_evaluation_metrics()
    test_benchmark_runner()
    print("All Phase 4 Data Layer & Training tests passed successfully!")
