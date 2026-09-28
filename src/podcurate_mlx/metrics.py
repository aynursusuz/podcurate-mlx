"""Text agreement is an ASR proxy, not measured transcription ground truth."""

import math
import unicodedata

from rapidfuzz.distance import Levenshtein

LANGUAGES = {"en", "zh", "ja", "tr"}
NUMERIC_METRICS = {
    "duration_s",
    "rms_dbfs",
    "peak",
    "full_scale_ratio",
    "avg_logprob",
    "no_speech_prob",
    "compression_ratio",
    "reference_cer",
    "reference_wer",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "overlap_ratio",
    "diar_speakers",
    "speaker_similarity",
    "speaker_consistency",
    "alignment_coverage",
    "alignment_score",
    "alignment_token_score",
}


def normalize(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    if language == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower()
    # Preserve letters, numbers and combining marks. Punctuation becomes a separator.
    text = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in text)
    return " ".join(text.split())


def agreement(reference: str, hypothesis: str, language: str) -> dict:
    ref, hyp = normalize(reference, language), normalize(hypothesis, language)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref_chars:
        raise ValueError("reference_text is empty after normalization")
    return {
        "reference_cer": Levenshtein.distance(ref_chars, hyp_chars) / len(ref_chars),
        # Space-delimited WER would be misleading for Chinese and Japanese.
        "reference_wer": (
            Levenshtein.distance(ref.split(), hyp.split()) / len(ref.split())
            if language in {"en", "tr"}
            else None
        ),
    }


def validate_policy(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != {"languages"}:
        raise ValueError("policy must contain only a languages object")
    if not isinstance(policy["languages"], dict) or not policy["languages"]:
        raise ValueError("languages must be a nonempty mapping")
    for lang, bounds in policy["languages"].items():
        if lang not in LANGUAGES or not isinstance(bounds, dict) or not bounds:
            raise ValueError("each supported language needs explicit metric bounds")
        for metric, interval in bounds.items():
            if metric not in NUMERIC_METRICS or not isinstance(interval, dict):
                raise ValueError(f"unsupported metric: {metric}")
            if not interval or set(interval) - {"min", "max"}:
                raise ValueError(f"{metric}: use min and/or max")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in interval.values()
            ):
                raise ValueError(f"{metric}: bounds must be finite numbers")
            if interval.get("min", -math.inf) > interval.get("max", math.inf):
                raise ValueError(f"{metric}: min exceeds max")


def decision(record: dict, policy: dict) -> tuple[str, list[str]]:
    if record.get("error"):
        return "error", [record["error"]]
    if record.get("boundary_cut"):
        return "review", ["boundary_cut"]
    if record.get("silent"):
        return "reject", ["silent"]
    if not normalize(record.get("text", ""), record["language"]):
        return "reject", ["empty_transcript"]
    bounds = policy["languages"].get(record["language"])
    if not bounds:
        return "review", ["language_not_calibrated"]
    rejected, missing = [], []
    for metric, interval in bounds.items():
        value = record.get(metric)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            missing.append(f"{metric}:missing")
        elif value < interval.get("min", -math.inf) or value > interval.get("max", math.inf):
            rejected.append(f"{metric}:outside_bounds")
    if rejected:
        return "reject", rejected + missing
    return ("review", missing) if missing else ("accept", [])


def validate_profiles(policy):
    if set(policy) != {"profiles"} or not isinstance(policy["profiles"], dict):
        raise ValueError("profile policy must contain only profiles")
    if not policy["profiles"] or set(policy["profiles"]) - {"tts", "asr"}:
        raise ValueError("profiles must contain tts and/or asr")
    for profile in policy["profiles"].values():
        if not isinstance(profile, dict) or set(profile) != {"languages"}:
            raise ValueError("each profile needs languages")
        for language, kinds in profile["languages"].items():
            if not isinstance(kinds, dict) or not kinds or set(kinds) - {"podcast", "synthetic"}:
                raise ValueError("each language needs podcast and/or synthetic bounds")
            for bounds in kinds.values():
                validate_policy({"languages": {language: bounds}})


def profile_decision(record, policy, profile):
    """Structural requirements plus explicit user bounds; no universal MOS/cosine cutoffs."""
    if profile not in {"tts", "asr"}:
        raise ValueError("profile must be tts or asr")
    kind = record.get("kind", record.get("input", {}).get("kind", "podcast"))
    bounds = (
        policy["profiles"]
        .get(profile, {})
        .get("languages", {})
        .get(record["language"], {})
        .get(kind)
    )
    if record.get("error"):
        return "error", [record["error"]]
    if record.get("silent"):
        return "reject", ["silent"]
    if not bounds:
        return "review", ["language_source_profile_not_calibrated"]
    required = {"asr"}
    required.add("alignment_tr" if record["language"] == "tr" else "alignment_qwen")
    if profile == "tts":
        required.update({"diarization", "speaker", "quality"})
    missing = [
        f"{name}:missing_stage"
        for name in sorted(required)
        if record.get("stages", {}).get(name, {}).get("status") != "ok"
    ]
    if record.get("alignment_status") != "ok":
        missing.append("alignment_requires_review")
    if record.get("boundary_cut"):
        missing.append("boundary_cut")
    if record.get("input", {}).get("reference_text"):
        if not ({"reference_cer", "reference_wer"} & bounds.keys()):
            missing.append("reference_agreement_bound_required")
    elif not {"avg_logprob", "compression_ratio"} <= bounds.keys():
        missing.append("untranscribed_asr_bounds_required")
    if profile == "tts":
        if record.get("diar_speakers") != 1:
            missing.append("tts_requires_one_speaker")
        if record.get("speaker_capacity_reached"):
            missing.append("diarization_capacity_reached")
        if record.get("speaker_status") != "ok":
            missing.append("speaker_requires_review")
        for metric in ("overlap_ratio", "speaker_consistency", "dnsmos_ovrl"):
            if metric not in bounds:
                missing.append(f"{metric}:bound_required")
        if record.get("speaker_reference_available") and "speaker_similarity" not in bounds:
            missing.append("speaker_similarity:bound_required")
    effective_bounds = dict(bounds)
    if profile == "tts" and not record.get("speaker_reference_available"):
        effective_bounds.pop("speaker_similarity", None)
    status, reasons = decision(record, {"languages": {record["language"]: effective_bounds}})
    if status in {"reject", "error"}:
        return status, reasons + missing
    return ("review", reasons + missing) if reasons or missing else ("accept", [])
