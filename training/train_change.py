"""
Training script for Bi-Temporal Change Detection Model
"""

import torch
import torch.nn as nn
from training.data_loaders import create_dataloader
from training.trainer import BaseTrainer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleChangeNet(nn.Module):
    def __init__(self):
        super(SimpleChangeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.encoder(x)


def run_change_training(epochs: int = 2, batch_size: int = 4):
    logger.info("Initializing Change Detection model training pipeline...")
    train_loader = create_dataloader('CDVQA', split='train', batch_size=batch_size)
    val_loader = create_dataloader('CDVQA', split='val', batch_size=batch_size)

    net = SimpleChangeNet()
    trainer = BaseTrainer(net, train_loader, val_loader)
    result = trainer.train(num_epochs=epochs, model_name="change_detection_model")
    logger.info(f"Change Detection Training Complete: {result}")
    return result


if __name__ == '__main__':
    run_change_training()
