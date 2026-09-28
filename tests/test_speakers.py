from types import SimpleNamespace

import numpy as np
import pytest

from podcurate_mlx.speakers import ECAPASpeaker, NemotronDiarizer, _cosine


class FakeModel:
    def __init__(self):
        self.calls = []

    def init_streaming_state(self):
        return SimpleNamespace(frames_processed=0, samples=0)

    def feed(self, audio, state, **kwargs):
        self.calls.append((state, kwargs["final"]))
        state.samples += len(audio)
        if kwargs["final"]:
            rows = state.samples // 160 - state.frames_processed
            probs = np.zeros((rows, 8), np.float32)
            probs[:, 0] = 0.9
            probs[2:4, 1] = 0.8
            state.frames_processed += rows
        else:
            probs = np.zeros((0, 8), np.float32)
        return SimpleNamespace(speaker_probs=probs), state


def diarizer():
    instance = object.__new__(NemotronDiarizer)
    instance._model = FakeModel()
    instance._mx = SimpleNamespace(eval=lambda _: None)
    instance.frame_samples = 160
    instance.activity_threshold = 0.5
    return instance


def test_stream_preserves_state_flushes_once_and_is_lazy():
    instance = diarizer()
    consumed = []

    def blocks():
        for index in range(3):
            consumed.append(index)
            yield np.zeros(320, np.float32)

    output = instance.stream(blocks())
    assert consumed == []
    rows = list(output)
    assert len(rows) == 1
    assert rows[0]["start_sample"] == 0
    assert rows[0]["frame_samples"] == 160
    assert rows[0]["probabilities"].shape == (6, 8)
    assert [final for _, final in instance._model.calls] == [False, False, False, True]
    assert len({id(state) for state, _ in instance._model.calls}) == 1


def test_overlap_counts_simultaneous_channels_not_argmax():
    result = diarizer()(np.zeros(1600, np.float32))
    assert result["diar_speakers"] == 2
    assert result["overlap_ratio"] == pytest.approx(0.2)
    assert result["speaker_turns"] == [
        {"start": 0.0, "end": 0.1, "speaker": 0},
        {"start": 0.02, "end": 0.04, "speaker": 1},
    ]
    assert result["speaker_capacity_reached"] is False


def test_stream_empty_and_oversized_blocks():
    assert list(diarizer().stream([])) == []
    with pytest.raises(ValueError, match="30 seconds"):
        list(diarizer().stream([np.zeros(480001, np.float32)]))


def speaker():
    instance = object.__new__(ECAPASpeaker)
    instance.embed = lambda audio: np.full(192, 1 / np.sqrt(192), dtype=np.float32)
    return instance


def test_speaker_metrics_do_not_claim_reference_when_absent():
    result = speaker()(np.zeros(96000, np.float32))
    assert result["speaker_similarity"] is None
    assert result["speaker_reference_available"] is False
    assert result["speaker_consistency"] == pytest.approx(1)
    assert result["speaker_consistency_windows"] == 2
    assert result["speaker_status"] == "ok"


def test_short_speaker_is_review_without_fabricated_consistency():
    result = speaker()(np.zeros(16000, np.float32))
    assert result["speaker_consistency"] is None
    assert result["speaker_status"] == "review"


def test_two_second_speaker_uses_disjoint_halves():
    assert speaker()(np.zeros(32000, np.float32))["speaker_consistency_windows"] == 2


def test_cached_reference_and_invalid_reference():
    instance = speaker()
    result = instance(np.zeros(96000, np.float32), reference_embedding=np.ones(192))
    assert result["speaker_similarity"] == pytest.approx(1)
    assert result["speaker_reference_available"]
    with pytest.raises(ValueError, match="192"):
        instance(np.zeros(32000, np.float32), reference_embedding=np.ones(2))
    with pytest.raises(RuntimeError, match="invalid"):
        _cosine(np.ones(192), np.zeros(192))
    with pytest.raises(ValueError, match="not both"):
        instance(np.zeros(32000, np.float32), np.ones(32000), reference_embedding=np.ones(192))


def test_stream_offsets_are_captured_before_state_mutation():
    instance = diarizer()

    def feed(audio, state, **kwargs):
        rows = len(audio) // 160
        state.frames_processed += rows
        return SimpleNamespace(speaker_probs=np.zeros((rows, 8), np.float32)), state

    instance._model.feed = feed
    rows = list(instance.stream([np.zeros(320, np.float32), np.zeros(480, np.float32)]))
    assert [row["start_sample"] for row in rows] == [0, 320]
