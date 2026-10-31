from __future__ import annotations

import json
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_recall_fscore_support)

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import deduplicate, load_dataset, split_data, validate_dataframe
from inf8239_u02.modeling import build_models

CLASSES = ["negative", "positive"]


def main() -> None:
    T, Y = settings.text_column, settings.target_column
    df = load_dataset().dropna(subset=[T, Y])
    validate_dataframe(df, T, Y)
    df, dedup = deduplicate(df, T, Y)
    print("Deduplicación antes de dividir:", dedup)
    train, test = split_data(df, T, Y, random_state=settings.random_state)
    overlap = set(train[T].str.lower()) & set(test[T].str.lower())
    assert not overlap, "Hay textos repetidos entre entrenamiento y prueba"
    print(f"Entrenamiento: {len(train)} · Prueba: {len(test)} · Textos compartidos: {len(overlap)}")

    reports, models_dir = ROOT / "reports", ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    rows, predictions, fitted = [], {}, {}
    for name, model in build_models(settings.random_state).items():
        start = perf_counter(); model.fit(train[T], train[Y]); fit_s = perf_counter() - start
        start = perf_counter(); pred = model.predict(test[T]); pred_ms = (perf_counter() - start) * 1000
        predictions[name], fitted[name] = pred, model
        p, r, f, s = precision_recall_fscore_support(test[Y], pred, labels=CLASSES, zero_division=0)
        row = {"model": name, "f1_macro": f1_score(test[Y], pred, average="macro"),
               "accuracy": accuracy_score(test[Y], pred), "fit_seconds": fit_s, "predict_ms": pred_ms,
               "vocabulary": len(model.named_steps["tfidf"].vocabulary_)}
        for i, c in enumerate(CLASSES):
            row.update({f"precision_{c}": p[i], f"recall_{c}": r[i], f"f1_{c}": f[i]})
        rows.append(row)
        print(f"\n{name} · F1 macro={row['f1_macro']:.3f}")
        print(classification_report(test[Y], pred, labels=CLASSES, digits=3, zero_division=0))

    metrics = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    metrics.to_csv(reports / "text_metrics.csv", index=False)

    # Métricas por fuente (amazon, imdb, yelp) para cada modelo
    by_source = []
    for name, pred in predictions.items():
        frame = test.assign(pred=pred)
        for source, g in frame.groupby("source"):
            by_source.append({"model": name, "source": source, "n": len(g),
                              "f1_macro": f1_score(g[Y], g["pred"], average="macro"),
                              "accuracy": accuracy_score(g[Y], g["pred"])})
    pd.DataFrame(by_source).to_csv(reports / "text_metrics_by_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["naive_bayes", "logistic"]):
        ConfusionMatrixDisplay.from_predictions(test[Y], predictions[name], labels=CLASSES,
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(reports / "confusion_text.png", dpi=170)

    selected_name = metrics[metrics["model"] != "dummy"].iloc[0]["model"]
    selected, selected_pred = fitted[selected_name], predictions[selected_name]
    errors = test.assign(predicted=selected_pred)
    errors = errors[errors[Y] != errors["predicted"]].rename(columns={Y: "real"})
    if hasattr(selected, "predict_proba"):
        proba = selected.predict_proba(errors[T])
        errors["confidence"] = proba.max(axis=1).round(3)
    errors["category"] = "REVISAR"
    errors[[T, "source", "real", "predicted", "confidence", "category"]].to_csv(
        reports / "error_analysis.csv", index=False)
    sample = errors.sample(min(20, len(errors)), random_state=settings.random_state)
    sample[[T, "source", "real", "predicted", "confidence", "category"]].to_csv(
        reports / "error_analysis_20.csv", index=False)

    joblib.dump(selected, models_dir / "text_model.joblib")
    (reports / "selected_model.json").write_text(json.dumps(
        {"selected": selected_name, "train_rows": len(train), "test_rows": len(test),
         "deduplication": dedup, "errors_total": int(len(errors))}, indent=2), encoding="utf-8")
    print(f"\nModelo seleccionado: {selected_name} · errores en prueba: {len(errors)}")
    print("Artefactos guardados en reports/ y models/")


if __name__ == "__main__":
    main()
