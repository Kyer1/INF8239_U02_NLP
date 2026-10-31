# Declaración de uso de IA · Ejercicio 03

- **Herramienta:** Claude (Anthropic), asistente conversacional.
- **Para qué se usó:**
  - Comparar candidatos de corpus y verificar licencia y procedencia en la página de UCI.
  - Adaptar `download_data.py`, `audit_data.py` y `train_text.py` del proyecto base para leer el ZIP de UCI, deduplicar antes de dividir, estratificar por fuente y guardar métricas por clase y por fuente.
  - Proponer pruebas (`test_corpus.py`, `test_evaluation.py`).
  - Redactar un primer borrador de la ficha del dataset, de las categorías de los 20 errores y de las interpretaciones.
- **Prompts relevantes:** «Busca 2 datasets que me funcionen para terminar estas tareas»; «Ayúdame con el Ejercicio 03 usando el proyecto base del profesor»; «Explícame paso a paso cómo ejecutar uv, Git y las pruebas».
- **Partes verificadas por mí:**
  - Ejecuté la descarga, la auditoría, el entrenamiento, el notebook y `uv run pytest -q` en mi computadora.
  - Comprobé el SHA-256 y que la auditoría coincide con la ficha (3,000 oraciones, 21 duplicados, clases balanceadas).
  - Revisé una por una las 20 oraciones mal clasificadas y su categoría en `docs/error_review.csv`.
  - Confirmé la licencia CC BY 4.0 y la cita en la página oficial del dataset.
- **Correcciones realizadas:**
  - Al leer el archivo de IMDb aparecían filas cortadas porque algunas oraciones contienen caracteres que se interpretan como salto de línea; se cambió la lectura para dividir solo por `\n` y se agregó una prueba.
  - Un error tenía un carácter especial (–) que impedía unirlo con su categoría; se creó `text_key` para comparar solo letras y números.
