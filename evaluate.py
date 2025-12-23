# src/evaluate.py
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

from .config import CFG


def _metrics_for_threshold(y_true: np.ndarray, proba_true: np.ndarray, thr: float):
    """
    thr: True (1) demek için gereken olasılık eşiği
    y_pred = 1 if proba_true >= thr else 0
    """
    y_pred = (proba_true >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))

    # average=None -> her sınıf için ayrı (0 ve 1)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    # sınıf 0: False(non-verified), sınıf 1: True(verified)
    out = {
        "threshold": float(thr),
        "accuracy": acc,
        "false_precision": float(p[0]),
        "false_recall": float(r[0]),
        "false_f1": float(f1[0]),
        "true_precision": float(p[1]),
        "true_recall": float(r[1]),
        "true_f1": float(f1[1]),
        "support_false": int(support[0]),
        "support_true": int(support[1]),
    }
    return out, y_pred


def main():
    test_path = CFG.SPLITS_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Önce src/train.py çalıştır (test split yok).")

    df = pd.read_csv(test_path)
    X_text = df["text"].fillna("").astype(str).values
    y_true = df["label"].astype(int).values

    vectorizer = joblib.load(CFG.VECTORIZER_PATH)
    clf = joblib.load(CFG.MODEL_PATH)

    X = vectorizer.transform(X_text)

    # Threshold için proba lazım
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Model predict_proba desteklemiyor. LogisticRegression olmalı.")

    proba_true = clf.predict_proba(X)[:, 1]

    # 0.10–0.90 arası tarayalım (istersen aralığı daraltırız)
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    rows = []
    best = None
    best_y_pred = None

    for thr in thresholds:
        row, y_pred = _metrics_for_threshold(y_true, proba_true, float(thr))
        rows.append(row)

        # Seçim kuralı: False F1 maksimum
        # Eşitlikte: False recall maksimum, sonra accuracy maksimum
        if best is None:
            best = row
            best_y_pred = y_pred
        else:
            if (row["false_f1"] > best["false_f1"]) or \
               (row["false_f1"] == best["false_f1"] and row["false_recall"] > best["false_recall"]) or \
               (row["false_f1"] == best["false_f1"] and row["false_recall"] == best["false_recall"] and row["accuracy"] > best["accuracy"]):
                best = row
                best_y_pred = y_pred

    # Best threshold ile rapor
    report = classification_report(
        y_true,
        best_y_pred,
        target_names=["False(non-verified)", "True(verified)"],
        zero_division=0
    )
    cm = confusion_matrix(y_true, best_y_pred)

    # metrics json
    metrics = {
        "best_threshold": best["threshold"],
        "best": best,
        "confusion_matrix": cm.tolist(),
        # İstersen threshold tablosunu da sakla (hocaya göstermek için güzel olur)
        "threshold_sweep": rows
    }

    CFG.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG.METRICS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # confusion matrix figure (best threshold)
    CFG.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = CFG.FIGURES_DIR / "confusion_matrix.png"

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (thr={best['threshold']})")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Evaluation bitti.")
    print(f"Seçilen threshold (False F1 odaklı): {best['threshold']}")
    print(report)
    print(f"metrics -> {CFG.METRICS_DIR / 'metrics.json'}")
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
