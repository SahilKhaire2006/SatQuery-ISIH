from typing import List, Dict, Any


def compute_change_metrics(
    pred_changes: List[float],
    gt_changes: List[float],
    tolerance: float = 10.0
) -> Dict[str, float]:
    """
    Compute change detection performance metrics.
    """
    if not pred_changes or not gt_changes:
        return {'mae': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}

    errors = []
    tp = 0

    for pred, gt in zip(pred_changes, gt_changes):
        err = abs(pred - gt)
        errors.append(err)
        if err <= tolerance:
            tp += 1

    total = len(pred_changes)
    mae = sum(errors) / total
    precision = tp / total

    return {
        'mae': round(mae, 4),
        'precision': round(precision, 4),
        'recall': round(precision, 4),
        'f1_score': round(precision, 4)
    }
