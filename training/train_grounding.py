"""
Training script for Object Localization & Grounding Model
"""

import torch
import torch.nn as nn
from training.data_loaders import create_dataloader
from training.trainer import BaseTrainer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleGroundingNet(nn.Module):
    def __init__(self):
        super(SimpleGroundingNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.conv(x)


def run_grounding_training(epochs: int = 2, batch_size: int = 4):
    logger.info("Initializing Grounding model training pipeline...")
    train_loader = create_dataloader('CARTOSAT', split='train', batch_size=batch_size)
    val_loader = create_dataloader('CARTOSAT', split='val', batch_size=batch_size)

    net = SimpleGroundingNet()
    trainer = BaseTrainer(net, train_loader, val_loader)
    result = trainer.train(num_epochs=epochs, model_name="grounding_model")
    logger.info(f"Grounding Training Complete: {result}")
    return result


if __name__ == '__main__':
    run_grounding_training()
