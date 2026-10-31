# INF-8239 · Unidad 02 · Ejercicio 03 — Clasificador de texto reproducible

**Autora:** Kimberly Pereyra · Proyecto base: Edwin Ramón José Nolasco

Clasificación de sentimiento (negative / positive) de oraciones de reseñas con TF-IDF, comparando un baseline y dos pipelines (Complement Naive Bayes y Regresión logística).

## Corpus
- *Sentiment Labelled Sentences* — UCI ML Repository, id 331 (Kotzias, 2015). DOI 10.24432/C57604. Licencia **CC BY 4.0**.
- 3,000 oraciones en inglés (Amazon, IMDb, Yelp; 500 positivas y 500 negativas por fuente).
- Ficha completa: `docs/DATASET_CARD.md` · Candidatos comparados: `docs/dataset_candidates.csv`.

## Instalación (Windows, macOS o Linux)
```bash
uv python install 3.12
uv sync
```
Copie `.env.example` como `.env` (ya trae la URL oficial del corpus).

## Ejecución completa
```bash
uv run python scripts/download_data.py   # descarga el ZIP oficial de UCI e imprime SHA-256
uv run python scripts/audit_data.py      # nulos, duplicados, clases, longitudes → reports/audit_*
uv run python scripts/train_text.py      # baseline + NB + logística → reports/ y models/
uv run pytest -q                         # 13 pruebas
```
Notebook ejecutado: `notebooks/ex03_clasificacion_texto.ipynb` (kernel `.venv`).
Aplicación local opcional: `uv run streamlit run app/streamlit_app.py`.

## Decisiones principales
- Duplicados (21 tras normalizar) eliminados **antes** de dividir; textos con etiquetas contradictorias se descartarían.
- Partición única 75/25 estratificada por clase y fuente, semilla 42; el script comprueba que no haya textos compartidos.
- TF-IDF dentro de cada `Pipeline`: el vocabulario se aprende solo con entrenamiento.
- Métrica principal: F1 macro. Análisis de 20 errores categorizados en `docs/error_review.csv` → `reports/error_analysis_20_reviewed.csv`.

## Resultados
Ver `reports/text_metrics.csv`, `reports/text_metrics_by_source.csv`, `reports/confusion_text.png` y `reports/error_categories.png`.

## Estructura
```
data/raw/        corpus descargado (no se versiona)
docs/            ficha, candidatos, revisión de errores, uso de IA
notebooks/       notebook del ejercicio
reports/         auditoría, métricas, figuras, errores
models/          modelo seleccionado (text_model.joblib)
src/inf8239_u02/ carga, auditoría, deduplicación, partición, modelos, evaluación
scripts/         descarga, auditoría, entrenamiento
tests/           contrato de datos, parser del corpus, partición sin fuga, modelos
```

## Dependencias
`uv.lock` fija las versiones. Exportadas también en `requirements.txt` (pip) y `requirements-cloud.txt`.
