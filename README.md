# pronounce-assess

A simple-to-use, real-time voice pronunciation assessment library for Python.

https://github.com/user-attachments/assets/d95f2955-d778-4ab6-8f20-2872627a4e2c

## Installation

```bash
pip install pronounce-assess   # numpy, torch, transformers, sounddevice, eng-to-ipa
```
**Note:** For real-time voice assessment, this project is best used with GPU. If you are on CPU, the assessment will still work, but may be much slower.

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


## Development

```bash
# Install in editable mode with dev tools
pip install -e .[dev]

# Run tests
pytest
```

## License

MIT
