import pandas as pd

from inf8239_u02.evaluation import merge_error_review, text_key


def test_text_key_ignores_case_spaces_and_symbols():
    assert text_key("It's  GOOD – really!") == text_key("its good really")


def test_merge_marks_unreviewed_errors():
    errors = pd.DataFrame({"text": ["Bad phone.", "Nice!"], "real": ["negative", "positive"]})
    review = pd.DataFrame({"text": ["bad phone"], "category": ["negación"], "justification": ["x"]})
    merged = merge_error_review(errors, review)
    assert list(merged["category"]) == ["negación", "SIN REVISAR"]
