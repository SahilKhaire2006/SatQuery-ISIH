from typing import List, Dict, Any


def compute_vqa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute VQA evaluation metrics: Accuracy, Exact Match, Keyword F1.
    """
    if not predictions or not references:
        return {'accuracy': 0.0, 'exact_match': 0.0, 'f1_score': 0.0}

    total = len(predictions)
    exact_matches = 0
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_clean = str(pred).strip().lower()
        ref_clean = str(ref).strip().lower()

        if pred_clean == ref_clean:
            exact_matches += 1

        pred_tokens = set(pred_clean.split())
        ref_tokens = set(ref_clean.split())

        intersection = pred_tokens.intersection(ref_tokens)
        if not pred_tokens or not ref_tokens:
            f1 = 1.0 if pred_tokens == ref_tokens else 0.0
        else:
            precision = len(intersection) / len(pred_tokens)
            recall = len(intersection) / len(ref_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    exact_match_ratio = exact_matches / total
    avg_f1 = sum(f1_scores) / total

    return {
        'accuracy': round(avg_f1, 4),
        'exact_match': round(exact_match_ratio, 4),
        'keyword_f1': round(avg_f1, 4)
    }
