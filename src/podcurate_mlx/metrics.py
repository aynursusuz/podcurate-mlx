"""Text agreement is an ASR proxy, not measured transcription ground truth."""

import math
import unicodedata

from rapidfuzz.distance import Levenshtein

LANGUAGES = {"en", "zh", "ja", "tr"}
NUMERIC_METRICS = {
    "duration_s", "rms_dbfs", "peak", "full_scale_ratio", "avg_logprob", "no_speech_prob",
    "compression_ratio", "reference_cer", "reference_wer", "dnsmos_sig", "dnsmos_bak",
    "dnsmos_ovrl",
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
        "reference_wer": (Levenshtein.distance(ref.split(), hyp.split()) / len(ref.split())
                          if language in {"en", "tr"} else None),
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
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in interval.values()):
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
        if (value is None or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            missing.append(f"{metric}:missing")
        elif value < interval.get("min", -math.inf) or value > interval.get("max", math.inf):
            rejected.append(f"{metric}:outside_bounds")
    if rejected:
        return "reject", rejected + missing
    return ("review", missing) if missing else ("accept", [])
