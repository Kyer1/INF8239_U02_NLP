from __future__ import annotations

import sys

import requests

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import corpus_from_zip, sha256


def main() -> int:
    if settings.data_source == "local":
        print(f"Modo local. Archivo configurado: {settings.dataset_path}")
        return 0
    if settings.data_source != "url" or not settings.dataset_url:
        print("Configure DATA_SOURCE=url y DATASET_URL en .env", file=sys.stderr)
        return 2
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(settings.dataset_url, timeout=60)
    response.raise_for_status()
    if settings.dataset_url.lower().endswith(".zip"):
        source_zip = raw_dir / "source.zip"
        source_zip.write_bytes(response.content)
        print(f"ZIP original: {source_zip}")
        print(f"SHA-256 ZIP: {sha256(source_zip)}")
        corpus = corpus_from_zip(response.content)
        corpus.to_csv(settings.dataset_path, index=False)
    else:
        settings.dataset_path.write_bytes(response.content)
    print(f"Guardado: {settings.dataset_path}")
    print(f"SHA-256 CSV: {sha256(settings.dataset_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
