# pronounce-assess

A simple-to-use, real-time voice pronunciation assessment library for Python.

This library was created because there was a lack of such open-source tools. The only similar tools that inspired this library are the [Azure AI Foundry API](https://ai.azure.com/explore/aiservices/speech/pronunciationassessment) and [SpeechSuper](https://www.speechsuper.com/).

https://github.com/user-attachments/assets/d95f2955-d778-4ab6-8f20-2872627a4e2c

If you would like any feature not currently included, I would be happy to make it for you! [Suggestion form](https://forms.gle/Bc9WRUeAaS2VTRmU7)

## Installation

```bash
pip install pronounce-assess
```
**Update:** This library works for both GPU and CPU. Both are real-time, and the inference latency for cpu is ~240 ms.

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
it audio from a file, a websocket, or anything else.
See [examples/live_mic_demo.py](examples/live_mic_demo.py) for a live-microphone demo.
For a gradio demo, clone the project and run 
```
uv run .\examples\gradio_demo.py
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
