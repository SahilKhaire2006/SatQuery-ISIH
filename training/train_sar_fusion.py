"""
Training script for Optical-SAR Multi-Modal Fusion Model (USP-1)
"""

import torch
import torch.nn as nn
from training.data_loaders import create_dataloader
from training.trainer import BaseTrainer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleSARFusionNet(nn.Module):
    def __init__(self):
        super(SimpleSARFusionNet, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fusion(x)


def run_sar_fusion_training(epochs: int = 2, batch_size: int = 4):
    logger.info("Initializing Optical-SAR Fusion model training pipeline (USP-1)...")
    train_loader = create_dataloader('ISRO_SAC', split='train', batch_size=batch_size)
    val_loader = create_dataloader('ISRO_SAC', split='val', batch_size=batch_size)

    net = SimpleSARFusionNet()
    trainer = BaseTrainer(net, train_loader, val_loader)
    result = trainer.train(num_epochs=epochs, model_name="sar_fusion_model")
    logger.info(f"SAR Fusion Training Complete: {result}")
    return result


if __name__ == '__main__':
    run_sar_fusion_training()
