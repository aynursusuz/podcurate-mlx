"""Conservative selection contracts; thresholds here are test fixtures, not calibration."""

import pytest

from podcurate_mlx.metrics import agreement, decision, normalize, validate_policy


def test_turkish_case_mapping_and_unicode_normalization():
    assert normalize("I İ ı i; ISPARTA İZMİR", "tr") == "ı i ı i ısparta izmir"
    assert agreement("İYİ BİR IŞIK", "iyi bir ışık", "tr") == {
        "reference_cer": 0.0, "reference_wer": 0.0,
    }
    assert normalize("ＡＢＣ café", "en") == "abc café"


@pytest.mark.parametrize("language,reference,hypothesis,cer", [
    ("zh", "你好世界", "你好世", 0.25),
    ("ja", "こんにちは", "こんちは", 0.2),
])
def test_unsegmented_languages_report_cer_without_claiming_wer(
    language, reference, hypothesis, cer,
):
    result = agreement(reference, hypothesis, language)
    assert result["reference_cer"] == pytest.approx(cer)
    assert result["reference_wer"] is None


def test_word_errors_and_empty_normalized_reference():
    assert agreement("one two", "one three four", "en")["reference_wer"] == 1.0
    with pytest.raises(ValueError, match="empty"):
        agreement("?!", "hello", "en")


@pytest.mark.parametrize("score", [None, "0.1", float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_required_metric_is_review(score):
    status, reasons = decision(
        {"text": "hello", "language": "en", "reference_cer": score},
        {"languages": {"en": {"reference_cer": {"max": 0.2}}}},
    )
    assert (status, reasons) == ("review", ["reference_cer:missing"])


def test_boolean_is_not_a_numeric_quality_measure():
    status, _ = decision(
        {"text": "hello", "language": "en", "reference_cer": False},
        {"languages": {"en": {"reference_cer": {"max": 0.2}}}},
    )
    assert status == "review"


def test_errors_boundaries_and_uncalibrated_languages_cannot_be_accepted():
    policy = {"languages": {"en": {"reference_cer": {"max": 0.2}}}}
    base = {"text": "hello", "language": "en", "reference_cer": 0.0}
    assert decision({**base, "error": "decoder failed"}, policy)[0] == "error"
    assert decision({**base, "boundary_cut": True}, policy) == ("review", ["boundary_cut"])
    assert decision({**base, "boundary_cut": True, "reference_cer": 5.0}, policy) == (
        "review", ["boundary_cut"],
    )
    assert decision({**base, "language": "tr"}, policy) == (
        "review", ["language_not_calibrated"],
    )
    assert decision({**base, "silent": True}, policy) == ("reject", ["silent"])
    assert decision({**base, "text": "?!"}, policy) == ("reject", ["empty_transcript"])


@pytest.mark.parametrize("bounds", [
    {"max": float("nan")}, {"max": float("inf")}, {"max": True},
    {"min": 2, "max": 1}, {"maximum": 1}, {},
])
def test_invalid_policy_is_rejected(bounds):
    with pytest.raises(ValueError):
        validate_policy({"languages": {"en": {"reference_cer": bounds}}})
