from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings

# Archivos del corpus dentro del ZIP oficial de UCI y dominio de cada uno
SOURCE_FILES = {
    "amazon_cells_labelled.txt": "amazon",
    "imdb_labelled.txt": "imdb",
    "yelp_labelled.txt": "yelp",
}
LABELS = {0: "negative", 1: "positive"}
_LINE = re.compile(r"^(.*\S)\s+([01])\s*$", flags=re.S)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labelled_lines(content: str) -> list[tuple[str, int]]:
    """Convierte líneas 'oración<TAB>0|1' en pares (texto, etiqueta).

    Se divide solo por '\\n' porque algunas oraciones de IMDb contienen otros
    caracteres de salto que pandas o splitlines() interpretan como filas nuevas.
    """
    rows = []
    for line in content.split("\n"):
        if not line.strip():
            continue
        match = _LINE.match(line)
        if not match:
            raise ValueError(f"Línea con formato inesperado: {line[:60]!r}")
        text = re.sub(r"\s+", " ", match.group(1)).strip()
        rows.append((text, int(match.group(2))))
    return rows


def corpus_from_zip(raw_zip: bytes) -> pd.DataFrame:
    """Une los tres archivos del ZIP en un DataFrame con text, label y source."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in SOURCE_FILES or "__MACOSX" in member:
                continue
            rows = parse_labelled_lines(archive.read(member).decode("utf-8", errors="replace"))
            frame = pd.DataFrame(rows, columns=["text", "label"])
            frame["label"] = frame["label"].map(LABELS)
            frame["source"] = SOURCE_FILES[name]
            frames.append(frame)
    if len(frames) != len(SOURCE_FILES):
        raise ValueError("El ZIP no contiene los tres archivos esperados del corpus")
    return pd.concat(frames, ignore_index=True)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    selected = path or settings.dataset_path
    if not selected.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {selected}. Ejecute: uv run python scripts/download_data.py"
        )
    return pd.read_csv(selected)


def validate_dataframe(df: pd.DataFrame, text_column: str, target_column: str) -> None:
    missing = {text_column, target_column} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El dataset está vacío")
    if df[text_column].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Existen textos vacíos")
    if df[target_column].nunique(dropna=True) < 2:
        raise ValueError("Se requieren al menos dos clases")


def normalize_text(text: str) -> str:
    """Clave de comparación para duplicados: minúsculas y espacios unificados."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def deduplicate(df: pd.DataFrame, text_column: str = "text", target_column: str = "label"):
    """Elimina duplicados antes de dividir para que un mismo texto no quede en train y test.

    - Textos repetidos con la misma etiqueta: se conserva la primera aparición.
    - Textos repetidos con etiquetas distintas: se eliminan todas sus apariciones (etiqueta ambigua).
    """
    key = df[text_column].map(normalize_text)
    labels_per_text = df.groupby(key)[target_column].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    keep = ~key.isin(conflicting)
    clean = df[keep].loc[~key[keep].duplicated()].reset_index(drop=True)
    summary = {
        "rows_before": int(len(df)),
        "duplicate_rows_same_label": int(key[keep].duplicated().sum()),
        "rows_with_conflicting_labels": int((~keep).sum()),
        "rows_after": int(len(clean)),
    }
    return clean, summary


def split_data(df: pd.DataFrame, text_column: str = "text", target_column: str = "label",
               test_size: float = 0.25, random_state: int = 42):
    """Partición única estratificada por clase y fuente, con semilla fija."""
    strata = df[target_column].astype(str) + "_" + df.get("source", "all").astype(str)
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=strata)
