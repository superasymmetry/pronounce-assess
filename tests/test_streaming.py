import numpy as np
import pytest

from openvoice import load_model, stream_decode

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def model_and_processor():
    processor, model = load_model("facebook/wav2vec2-base-960h")
    return processor, model


def test_silence_yields_no_matches(model_and_processor):
    processor, model = model_and_processor
    sample_rate = 16000
    silent_chunk = np.zeros(sample_rate, dtype=np.float32)
    reference_phonemes = ["h", "ə", "l", "oʊ"]

    results = list(stream_decode([silent_chunk], reference_phonemes,
                                 processor, model, "cpu", sample_rate))

    for event in results:
        assert event["label"] in {"correct", "mispronounced", "omitted", "insertion"}


def test_empty_stream_yields_nothing(model_and_processor):
    processor, model = model_and_processor
    results = list(stream_decode([], ["h", "ə"], processor, model))
    assert results == []
