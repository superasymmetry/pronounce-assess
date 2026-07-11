# pronounce-assess

A voice pronunciation assessment library for Python. Streams audio through a
wav2vec2 CTC phoneme model and yields per-phoneme match events
(`correct` / `mispronounced` / `omitted` / `insertion`) against a reference
IPA sequence.

## Installation

```bash
pip install pronounce-assess   # numpy, torch, transformers, sounddevice, eng-to-ipa
```

## Usage

```python
from pronounce_assess import PronounceAssessModel
from pronounce_assess.audio import ChunkRecord

assessor = PronounceAssessModel()  # loads the model; picks cuda/cpu automatically
reference = assessor.sentence_to_phonemes("The quick brown fox")

with ChunkRecord(duration=5, chunk_len=8000) as chunks:
    for event in assessor.stream_decode(chunks, reference):
        print(event["phoneme"], event["label"], event["score"])
```

The lower-level pieces (`load_model`, `sentence_to_phonemes`, `stream_decode`)
remain available if you want to manage the processor/model pair yourself.
`stream_decode` accepts any iterable of float32 numpy chunks, so you can feed
it audio from a file, a websocket, or anything else — see
[examples/live_mic_demo.py](examples/live_mic_demo.py) for a live-microphone demo.

## Project layout

```
pronounce_assess/
    __init__.py     public API + version
    streaming.py    stream_decode – the core alignment/scoring algorithm
    phonemes.py     IPA normalization, sentence → phoneme conversion
    models.py       wav2vec2 model/processor loading + PronounceAssessModel
    audio.py        microphone chunk generator (optional, needs sounddevice)
    exceptions.py   error types
tests/              pytest suite
examples/           runnable demo scripts
```

## Development

```bash
# Install in editable mode with dev tools
pip install -e .[dev]

# Run tests
pytest
```

## License

MIT
