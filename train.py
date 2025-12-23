# src/train.py
import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import CFG
from .preprocessing import load_and_prepare


def main():
    df = load_and_prepare()

    X_text = df["__text__"].values
    y = df["__label__"].astype(int).values  # False=0, True=1

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # N-gram güçlendirme (Non-Verified için kritik)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",     # dataset İngilizceyse iyi; Türkçeyse kaldır
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30000
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)  # train içinde kullanılmıyor ama debug için hazır

    clf = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        n_jobs=None
    )
    clf.fit(Xtr, y_train)

    CFG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, CFG.VECTORIZER_PATH)
    joblib.dump(clf, CFG.MODEL_PATH)

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "text_col": CFG.TEXT_COL,
        "label_col": CFG.LABEL_COL,
        "test_size": CFG.TEST_SIZE,
        "random_state": CFG.RANDOM_STATE,
        "model": "LogisticRegression(class_weight=balanced)",
        "vectorizer": "TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2, max_df=0.95)"
    }
    with open(CFG.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Splitleri kaydet
    import pandas as pd
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(CFG.SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(CFG.SPLITS_DIR / "test.csv", index=False)

    print("Training bitti.")
    print(f"Model: {CFG.MODEL_PATH}")
    print(f"Vectorizer: {CFG.VECTORIZER_PATH}")
    print(f"Splits: {CFG.SPLITS_DIR / 'train.csv'} , {CFG.SPLITS_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
