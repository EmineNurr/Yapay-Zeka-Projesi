# src/predict.py
import joblib

from .config import CFG
from .translate import tr_to_en


class VerifiedPurchasePredictor:
    def __init__(self, threshold: float | None = None):
        self.vectorizer = joblib.load(CFG.VECTORIZER_PATH)
        self.clf = joblib.load(CFG.MODEL_PATH)
        self.threshold = float(
            threshold if threshold is not None else CFG.DEFAULT_THRESHOLD
        )

    def predict(self, text: str):
        if not text or not text.strip():
            raise ValueError("Boş metin gönderildi.")

        # 🔥 TÜRKÇE → İNGİLİZCE
        text_en = tr_to_en(text)

        X = self.vectorizer.transform([text_en])
        proba_true = float(self.clf.predict_proba(X)[0, 1])
        pred = int(proba_true >= self.threshold)

        return pred, proba_true
