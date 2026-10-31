import io
import zipfile

import pandas as pd
import pytest

from inf8239_u02.data import corpus_from_zip, deduplicate, parse_labelled_lines, split_data


def _fake_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sentiment labelled sentences/amazon_cells_labelled.txt", "Great phone.\t1\nBad battery.\t0\n")
        archive.writestr("sentiment labelled sentences/imdb_labelled.txt", "A fine film.  \t1\nBoring plot.  \t0\n")
        archive.writestr("sentiment labelled sentences/yelp_labelled.txt", "Loved it.\t1\nCold food.\t0\n")
        archive.writestr("__MACOSX/sentiment labelled sentences/._yelp_labelled.txt", "basura")
    return buffer.getvalue()


def test_parse_keeps_text_and_label_with_trailing_spaces():
    assert parse_labelled_lines("Boring plot.  \t0\n\n") == [("Boring plot.", 0)]


def test_parse_rejects_line_without_label():
    with pytest.raises(ValueError):
        parse_labelled_lines("sin etiqueta")


def test_corpus_from_zip_has_three_sources_and_named_labels():
    df = corpus_from_zip(_fake_zip())
    assert list(df.columns) == ["text", "label", "source"]
    assert set(df["source"]) == {"amazon", "imdb", "yelp"}
    assert set(df["label"]) == {"negative", "positive"}


def test_deduplicate_removes_repeats_and_conflicts():
    df = pd.DataFrame({"text": ["Good", "good ", "Bad", "Meh", "meh"],
                       "label": ["positive", "positive", "negative", "positive", "negative"]})
    clean, summary = deduplicate(df)
    assert list(clean["text"]) == ["Good", "Bad"]
    assert summary["duplicate_rows_same_label"] == 1
    assert summary["rows_with_conflicting_labels"] == 2


def test_split_has_no_shared_texts():
    df = pd.DataFrame({"text": [f"texto {i}" for i in range(40)],
                       "label": ["positive", "negative"] * 20, "source": ["amazon"] * 40})
    train, test = split_data(df)
    assert not set(train["text"]) & set(test["text"])
    assert len(test) == 10
