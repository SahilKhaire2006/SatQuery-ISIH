"""
PyTorch Fine-Tuning Engine for Text-Guided Grounding Specialist Model

Fine-tunes open-vocabulary vision-language grounding projection layers on official DIOR-RSVG splits,
tracks standard RSVG metrics (Acc@0.5, Acc@0.7, mIoU), performs temperature calibration,
and saves reproducible model checkpoints.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from text_guided_grounding.dataset import RSVGDataset
from text_guided_grounding.model import TextGuidedGroundingModel
from text_guided_grounding.calibration import TemperatureScaler


def compute_box_iou(box1: list, box2: list) -> float:
    """Compute IoU between two bounding boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class GroundingAdapterNet(nn.Module):
    """
    Lightweight neural adaptation layer for fine-tuning text-guided grounding proposals
    on domain-specific satellite imagery features.
    """

    def __init__(self, in_features: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.box_refiner = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features)
        )
        self.logit_head = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, box_tensor: torch.Tensor, raw_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Residual bounding box refinement delta
        delta_box = self.box_refiner(box_tensor)
        refined_box = box_tensor + delta_box
        # Calibrated logit score
        combined = torch.cat([box_tensor, raw_score.unsqueeze(-1)], dim=-1)
        logit_score = self.logit_head(combined)
        return refined_box, logit_score.squeeze(-1)


def evaluate_split(
    model: TextGuidedGroundingModel,
    adapter: GroundingAdapterNet,
    dataset: RSVGDataset,
    scaler: TemperatureScaler
) -> Dict[str, float]:
    """
    Evaluate grounding metrics (Acc@0.5, Acc@0.7, mIoU) on a dataset split.
    """
    ious = []
    logits = []
    labels = []

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        query = sample["query"]
        gt_bbox = sample["gt_bbox"]

        res = model.predict(image, query, top_k=1)
        pred_box = res["bbox"]
        confidence = res["confidence"]

        # Refine box via adapter if available
        if adapter is not None:
            with torch.no_grad():
                b_in = torch.tensor([pred_box], dtype=torch.float32)
                s_in = torch.tensor([confidence], dtype=torch.float32)
                r_box, r_logit = adapter(b_in, s_in)
                pred_box = [float(x) for x in r_box[0].tolist()]

        iou = compute_box_iou(pred_box, gt_bbox)
        ious.append(iou)
        logits.append(confidence)
        labels.append(1 if iou >= 0.5 else 0)

    acc_05 = float(np.mean([1.0 if i >= 0.5 else 0.0 for i in ious])) if ious else 0.0
    acc_07 = float(np.mean([1.0 if i >= 0.7 else 0.0 for i in ious])) if ious else 0.0
    miou = float(np.mean(ious)) if ious else 0.0

    return {
        "Acc@0.5": round(acc_05, 4),
        "Acc@0.7": round(acc_07, 4),
        "mIoU": round(miou, 4),
        "logits": logits,
        "labels": labels
    }


def train_model(
    epochs: int = 5,
    lr: float = 1e-3,
    checkpoint_dir: Path = Path("text_guided_grounding/checkpoints")
) -> Dict[str, Any]:
    """
    Execute PyTorch fine-tuning loop on DIOR-RSVG dataset.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      TEXT-GUIDED GROUNDING MODEL — PHASE 2 FINE-TUNING")
    print("=" * 65)

    train_json = Path("text_guided_grounding/data/dior_rsvg/annotations/train.json")
    val_json = Path("text_guided_grounding/data/dior_rsvg/annotations/val.json")
    img_dir = Path("text_guided_grounding/data/dior_rsvg/images")

    train_ds = RSVGDataset(dataset_dir=img_dir, annotation_file=train_json, split="train")
    val_ds = RSVGDataset(dataset_dir=img_dir, annotation_file=val_json, split="val")

    print(f" Loaded Train Split : {len(train_ds)} samples")
    print(f" Loaded Val Split   : {len(val_ds)} samples")

    base_model = TextGuidedGroundingModel()
    adapter = GroundingAdapterNet()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    scaler = TemperatureScaler()

    best_miou = 0.0
    training_history = []

    print("\nStarting Fine-Tuning Epochs:")
    for epoch in range(1, epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        start_t = time.time()

        for i in range(len(train_ds)):
            sample = train_ds[i]
            image = sample["image"]
            query = sample["query"]
            gt_bbox = sample["gt_bbox"]

            res = base_model.predict(image, query, top_k=1)
            pred_box = res["bbox"]
            confidence = res["confidence"]

            b_in = torch.tensor([pred_box], dtype=torch.float32, requires_grad=True)
            s_in = torch.tensor([confidence], dtype=torch.float32)
            gt_tensor = torch.tensor([gt_bbox], dtype=torch.float32)

            r_box, r_logit = adapter(b_in, s_in)

            # Smooth L1 Box loss against ground truth
            loss = l1_loss(r_box, gt_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        elapsed = time.time() - start_t
        avg_loss = epoch_loss / max(1, len(train_ds))

        # Evaluate on validation split
        adapter.eval()
        val_metrics = evaluate_split(base_model, adapter, val_ds, scaler)
        scaler.fit(val_metrics["logits"], val_metrics["labels"])

        print(
            f" Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc@0.5: {val_metrics['Acc@0.5']:.2%} | "
            f"Val Acc@0.7: {val_metrics['Acc@0.7']:.2%} | "
            f"Val mIoU: {val_metrics['mIoU']:.4f} | "
            f"Temp T: {scaler.temperature:.3f}"
        )

        training_history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_acc_05": val_metrics["Acc@0.5"],
            "val_acc_07": val_metrics["Acc@0.7"],
            "val_miou": val_metrics["mIoU"],
            "temperature": scaler.temperature
        })

        if val_metrics["mIoU"] >= best_miou:
            best_miou = val_metrics["mIoU"]
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "adapter_state_dict": adapter.state_dict(),
                "temperature": scaler.temperature,
                "val_miou": best_miou,
                "epoch": epoch
            }, ckpt_path)

    # Save summary metadata
    meta_path = checkpoint_dir / "training_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "DIOR-RSVG",
            "epochs": epochs,
            "best_val_miou": best_miou,
            "final_temperature": scaler.temperature,
            "history": training_history
        }, f, indent=2)

    print("-" * 65)
    print(f" Training complete. Best Val mIoU: {best_miou:.4f}")
    print(f" Saved fine-tuned checkpoint to: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 65 + "\n")

    return {
        "best_val_miou": best_miou,
        "checkpoint_path": checkpoint_dir / "best_model.pt"
    }


if __name__ == "__main__":
    train_model(epochs=3)
