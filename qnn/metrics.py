from __future__ import annotations

from typing import List, Optional, Tuple


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    n = max(1, len(y_true))
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / n


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    # Map labels to {0,1} by assuming negatives are -1 or 0
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        yt01 = 1 if yt > 0 else 0
        yp01 = 1 if yp > 0 else 0
        if yt01 == 1 and yp01 == 1:
            tp += 1
        elif yt01 == 0 and yp01 == 1:
            fp += 1
        elif yt01 == 0 and yp01 == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def roc_auc(y_true: List[int], scores: List[float]) -> Optional[float]:
    # Convert y_true to {0,1}
    pairs = [(1 if yt > 0 else 0, s) for yt, s in zip(y_true, scores)]
    n_pos = sum(1 for yt, _ in pairs if yt == 1)
    n_neg = sum(1 for yt, _ in pairs if yt == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    # Rank scores (average ranks for ties)
    sorted_pairs = sorted(enumerate(pairs), key=lambda x: x[1][1])  # ascending by score
    ranks = [0.0] * len(pairs)
    i = 0
    rank = 1
    while i < len(sorted_pairs):
        j = i
        # tie group
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1][1] == sorted_pairs[i][1][1]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    sum_ranks_pos = sum(r for (yt, _), r in zip(pairs, ranks) if yt == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return p, r, f1


def pr_curve(y_true: List[int], scores: List[float], thresholds: int = 50) -> Tuple[List[float], List[float], List[float]]:
    ts = [min(scores) + (max(scores) - min(scores)) * i / max(1, thresholds - 1) for i in range(thresholds)]
    prec, rec, thr = [], [], []
    for t in ts:
        y_pred = [1 if s >= t else -1 for s in scores]
        p, r, _ = precision_recall_f1(y_true, y_pred)
        prec.append(p)
        rec.append(r)
        thr.append(t)
    return prec, rec, thr

