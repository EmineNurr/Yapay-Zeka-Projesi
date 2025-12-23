# src/preprocessing.py
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from .config import CFG


def _ensure_dirs():
    CFG.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CFG.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CFG.METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _read_any_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    data/raw içinden ilk uygun dosyayı bulup okur:
    - .csv
    - .json
    - .jsonl
    """
    candidates = []
    for ext in ("*.csv", "*.jsonl", "*.json"):
        candidates.extend(sorted(raw_dir.glob(ext)))

    if not candidates:
        raise FileNotFoundError(
            f"{raw_dir} içinde veri dosyası yok. CSV/JSON/JSONL bekliyorum."
        )

    path = candidates[0]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".json":
        # JSON dizi ya da sözlük olabilir
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        df = pd.DataFrame(obj) if isinstance(obj, list) else pd.DataFrame(obj)
    else:
        raise ValueError(f"Desteklenmeyen format: {path}")

    return df


def _safe_bool_series(s: pd.Series) -> pd.Series:
    """
    True/False, 1/0, 'true'/'false' gibi türleri normalize eder.
    """
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(int).astype(bool)

    # string normalize
    ss = s.astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f"}
    out = ss.map(lambda x: True if x in true_vals else (False if x in false_vals else np.nan))
    return out


def build_text(df: pd.DataFrame) -> pd.Series:
    text = df.get(CFG.TEXT_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    title = df.get(CFG.TITLE_COL, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    # Title + Text birleştir (title yoksa sorun değil)
    combined = (title.str.strip() + " " + text.str.strip()).str.strip()
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimum temizlik:
    - gerekli kolonları çek
    - boş metinleri at
    - label'ı bool'a çevir
    """
    needed = [CFG.LABEL_COL, CFG.TEXT_COL]
    # title/rating opsiyonel
    if CFG.TITLE_COL and CFG.TITLE_COL in df.columns:
        needed.append(CFG.TITLE_COL)
    if CFG.RATING_COL and CFG.RATING_COL in df.columns:
        needed.append(CFG.RATING_COL)

    missing = [c for c in [CFG.LABEL_COL, CFG.TEXT_COL] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset'te zorunlu kolon(lar) yok: {missing}. "
            f"Lütfen src/config.py içindeki kolon adlarını güncelle."
        )

    df2 = df[needed].copy()

    # text
    df2["__text__"] = build_text(df2)
    df2 = df2[df2["__text__"].str.len() > 0].copy()

    # label
    df2["__label__"] = _safe_bool_series(df2[CFG.LABEL_COL])
    df2 = df2[df2["__label__"].notna()].copy()
    df2["__label__"] = df2["__label__"].astype(bool)

    # rating opsiyonelse sayıya çevir
    if CFG.RATING_COL in df2.columns:
        df2["__rating__"] = pd.to_numeric(df2[CFG.RATING_COL], errors="coerce").fillna(-1)

    return df2


def load_and_prepare() -> pd.DataFrame:
    _ensure_dirs()
    df = _read_any_dataset(CFG.RAW_DIR)
    df = clean_dataframe(df)
    # Cache (istersen)
    out_path = CFG.PROCESSED_DIR / "dataset_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df

