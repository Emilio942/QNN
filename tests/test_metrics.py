from __future__ import annotations

from qnn.metrics import accuracy, confusion_counts, roc_auc


def test_metrics_basic():
    y = [1, -1, 1, -1]
    y_pred = [1, 1, -1, -1]
    acc = accuracy(y, y_pred)
    tp, fp, tn, fn = confusion_counts(y, y_pred)
    assert 0 <= acc <= 1
    assert tp + fp + tn + fn == len(y)

    scores = [0.9, 0.2, -0.1, -0.8]
    auc = roc_auc(y, scores)
    assert auc is None or (0.0 <= auc <= 1.0)
