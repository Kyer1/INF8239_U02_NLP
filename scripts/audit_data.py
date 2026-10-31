from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import deduplicate, load_dataset, normalize_text, sha256, validate_dataframe


def audit(df: pd.DataFrame) -> dict:
    text = df[settings.text_column].astype(str)
    lengths = text.str.split().str.len()
    _, dedup = deduplicate(df, settings.text_column, settings.target_column)
    return {
        "sha256_csv": sha256(settings.dataset_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": {c: int(v) for c, v in df.isna().sum().items()},
        "empty_texts": int(text.str.strip().eq("").sum()),
        "exact_duplicate_texts": int(text.duplicated().sum()),
        "normalized_duplicate_texts": int(text.map(normalize_text).duplicated().sum()),
        "deduplication": dedup,
        "class_distribution": df[settings.target_column].value_counts().to_dict(),
        "class_by_source": pd.crosstab(df["source"], df[settings.target_column]).to_dict()
        if "source" in df else {},
        "words_per_text": {k: float(round(v, 1)) for k, v in
                           lengths.describe(percentiles=[0.5, 0.9, 0.99]).items()},
    }


def main() -> None:
    df = load_dataset()
    validate_dataframe(df, settings.text_column, settings.target_column)
    result = audit(df)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "audit_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    lengths = df[settings.text_column].astype(str).str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pd.crosstab(df["source"], df[settings.target_column]).plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Clases por fuente", xlabel="Fuente", ylabel="Oraciones")
    axes[1].hist(lengths, bins=40)
    axes[1].set(title="Longitud de las oraciones", xlabel="Palabras", ylabel="Frecuencia")
    fig.tight_layout()
    fig.savefig(reports / "audit_corpus.png", dpi=160)
    print("Auditoría guardada en reports/audit_summary.json y reports/audit_corpus.png")


if __name__ == "__main__":
    main()
