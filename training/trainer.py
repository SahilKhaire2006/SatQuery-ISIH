import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """
    Extensible Base Trainer for SatQuery Model Fine-Tuning & Training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        device: Optional[str] = None,
        checkpoint_dir: str = './models/checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BaseTrainer initialized on device '{self.device}'. Checkpoint dir: {self.checkpoint_dir}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            # Dummy forward pass computation for trainer interface
            dummy_target = torch.ones((images.size(0), 1), device=self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, dummy_target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0


        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                dummy_target = torch.ones((images.size(0), 1), device=self.device)
                loss = self.criterion(outputs, dummy_target)
                total_loss += loss.item()
                batches += 1

        avg_val_loss = total_loss / max(1, batches)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self, num_epochs: int = 3, model_name: str = "satquery_model") -> Dict[str, Any]:
        logger.info(f"Starting training for {num_epochs} epochs...")
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, num_epochs + 1):
            t_loss = self.train_epoch(epoch)
            v_loss = self.validate()

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)

        # Save model checkpoint
        checkpoint_path = self.checkpoint_dir / f"{model_name}_latest.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': num_epochs,
            'history': history
        }, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        return {
            'epochs': num_epochs,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'checkpoint': str(checkpoint_path)
        }
