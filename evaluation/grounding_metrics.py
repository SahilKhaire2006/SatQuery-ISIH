from typing import List, Dict, Any


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return max(0.0, min(1.0, iou))


def compute_grounding_metrics(
    predicted_boxes: List[List[int]],
    ground_truth_boxes: List[List[int]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute mAP@50, Mean IoU, Precision, Recall for Object Localization.
    """
    if not predicted_boxes or not ground_truth_boxes:
        return {'mean_iou': 0.0, 'map50': 0.0, 'precision': 0.0, 'recall': 0.0}

    ious = []
    tp = 0

    for pred, gt in zip(predicted_boxes, ground_truth_boxes):
        iou = compute_iou(pred, gt)
        ious.append(iou)
        if iou >= iou_threshold:
            tp += 1

    total = len(predicted_boxes)
    mean_iou = sum(ious) / total
    precision = tp / total
    recall = tp / len(ground_truth_boxes)

    return {
        'mean_iou': round(mean_iou, 4),
        'map50': round(precision, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4)
    }
