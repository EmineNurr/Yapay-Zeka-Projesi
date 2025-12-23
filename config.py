# src/config.py
from pathlib import Path

class CFG:
    # Proje kökü
    ROOT = Path(__file__).resolve().parents[1]

    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SPLITS_DIR = DATA_DIR / "splits"

    MODELS_DIR = ROOT / "models"
    REPORTS_DIR = ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"
    METRICS_DIR = REPORTS_DIR / "metrics"

    # Dataset kolon adları (preprocessing bunları kullanıyor)
    TEXT_COL = "reviewText"
    TITLE_COL = "summary"
    RATING_COL = "overall"
    LABEL_COL = "verifiedPurchase"   # True/False

    # Train ayarları
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Model/vektör kayıt yolları
    VECTORIZER_PATH = MODELS_DIR / "tfidf.joblib"
    MODEL_PATH = MODELS_DIR / "clf.joblib"
    META_PATH = MODELS_DIR / "meta.json"

    # Demo/Inference: Non-Verified yakalamak için threshold (Evaluate bunu arar, yoksa kendisi seçecek)
    DEFAULT_THRESHOLD = 0.50
