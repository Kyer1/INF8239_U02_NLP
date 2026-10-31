from __future__ import annotations

import re

import pandas as pd


def text_key(text: str) -> str:
    """Clave robusta para emparejar textos: solo letras y números en minúscula."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def merge_error_review(errors: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Une los errores generados por el modelo con la categoría asignada en la revisión manual."""
    left = errors.drop(columns=["category"], errors="ignore").assign(_key=errors["text"].map(text_key))
    right = review.drop(columns=["text"]).assign(_key=review["text"].map(text_key))
    merged = left.merge(right, on="_key", how="left").drop(columns="_key")
    merged["category"] = merged["category"].fillna("SIN REVISAR")
    return merged
