# Dataset Card · Sentiment Labelled Sentences

## Identificación
- **Nombre:** Sentiment Labelled Sentences
- **Fuente original:** UCI Machine Learning Repository, id 331 — https://archive.ics.uci.edu/dataset/331
- **Descarga directa:** https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
- **Responsable:** Dimitrios Kotzias (donado el 29/05/2015). Creado para Kotzias, D., Denil, M., de Freitas, N. y Smyth, P. (2015). *From Group to Individual Labels using Deep Features*. KDD 2015.
- **Cita:** Kotzias, D. (2015). Sentiment Labelled Sentences [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604
- **Versión:** archivo ZIP publicado por UCI; se identifica con el hash SHA-256 que imprime `scripts/download_data.py` y que se guarda en `reports/audit_summary.json`.
- **Licencia:** Creative Commons Attribution 4.0 (CC BY 4.0): permite descargar, analizar y publicar resultados derivados citando la fuente.
- **Idioma:** inglés.

## Propósito y variable objetivo
- **Pregunta:** ¿Puede un clasificador basado en TF-IDF distinguir oraciones positivas y negativas de reseñas de productos, películas y restaurantes, y qué tipo de oraciones lo confunden?
- **Unidad de análisis:** una oración extraída de una reseña.
- **Variable objetivo:** `label` (`negative` / `positive`).
- **Uso previsto:** práctica académica de clasificación de texto y análisis de errores.

## Diccionario de datos

| Columna | Tipo | Descripción | Valores o unidad |
|---|---|---|---|
| `text` | texto | Oración de una reseña, sin modificar salvo unificar espacios en blanco | 1 a 71 palabras (mediana 10) |
| `label` | categórica | Sentimiento asignado por los autores (originalmente 0/1) | `negative` (0), `positive` (1) |
| `source` | categórica | Sitio del que proviene la oración (agregada a partir del nombre del archivo) | `amazon` (teléfonos y accesorios), `imdb` (películas), `yelp` (restaurantes) |

## Procedimiento de obtención
1. `uv run python scripts/download_data.py` descarga el ZIP oficial de UCI (URL en `.env`) y guarda una copia en `data/raw/source.zip`.
2. Se leen solo los archivos `amazon_cells_labelled.txt`, `imdb_labelled.txt` y `yelp_labelled.txt` (se ignoran `readme.txt`, `.DS_Store` y `__MACOSX`).
3. Cada línea tiene el formato `oración<TAB>0|1`. Se separa por `\n` únicamente, porque algunas oraciones de IMDb contienen caracteres que otras funciones interpretan como saltos de línea y producían filas incompletas.
4. Se convierte 0/1 en `negative`/`positive` y se agrega `source`. El resultado se guarda en `data/raw/dataset.csv` y se registra su SHA-256.
5. Los datos no se suben a Git; cualquiera los obtiene repitiendo el paso 1.

## Calidad observada
- 3,000 oraciones: 1,000 por fuente, 500 positivas y 500 negativas en cada una (clases balanceadas).
- 0 nulos y 0 textos vacíos.
- 18 duplicados exactos y 21 al normalizar mayúsculas y espacios, todos con la misma etiqueta (0 conflictos). Se eliminan antes de dividir, quedando 2,979 oraciones, para que un mismo texto no aparezca en entrenamiento y prueba.
- Textos muy cortos: el percentil 99 es de 36 palabras; hay oraciones de una sola palabra, lo que limita la información disponible.

## Ejemplos (sin datos personales)

| text | label | source |
|---|---|---|
| So there is no way for me to plug it in here in the US unless I go by a converter. | negative | amazon |
| Great for the jawbone. | positive | amazon |
| A very, very, very slow-moving, aimless movie about a distressed, drifting young man. | negative | imdb |
| Wow... Loved this place. | positive | yelp |
| Crust is not good. | negative | yelp |

## Población cubierta y excluida
- Cubre reseñas en inglés de tres sitios estadounidenses (Amazon, IMDb, Yelp), seleccionadas por los autores por tener sentimiento claro.
- **Excluye** oraciones neutrales o mixtas (fueron filtradas a propósito), otros idiomas y variantes (por ejemplo, español dominicano), redes sociales, mensajes largos y otros dominios. No se documenta la fecha exacta ni el perfil de quienes escribieron las reseñas.

## Riesgos, sesgos y usos prohibidos
- **Sesgo de selección:** al quitar oraciones neutrales, la tarea es más fácil que en datos reales; el desempeño no se trasladaría directamente.
- **Sesgo de dominio:** el vocabulario es específico de teléfonos, películas y restaurantes.
- **Etiquetado:** una sola etiqueta por oración, sin medida de acuerdo entre anotadores.
- **Usos prohibidos:** decidir sobre personas (empleados, clientes), moderar contenido de forma automática o aplicar el modelo a otros idiomas o dominios sin una nueva evaluación.
