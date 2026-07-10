"""Model loading helpers and the high-level OpenVoiceModel."""

DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"


def load_model(device, model_name=DEFAULT_MODEL):
    """Load a wav2vec2 CTC phoneme model and its processor.

    Returns:
        (processor, model) tuple, with the model in eval mode on ``device``.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
    model.to(device)
    return processor, model


class OpenVoiceModel:
    """Bundles the model, processor, and device behind a simple API.

    Loads the model once, then exposes the library's operations as methods
    so callers never handle the processor/model pair themselves::

        assessor = OpenVoiceModel()
        reference = assessor.sentence_to_phonemes("The quick brown fox")
        for event in assessor.stream_decode(chunks, reference):
            ...
    """

    def __init__(self, sentence=None, model_name=DEFAULT_MODEL, device=None):
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.processor, self.model = load_model(device, model_name)
        self.reference_phonemes = None
        if sentence is not None:
            self.set_sentence(sentence)

    def sentence_to_phonemes(self, sentence):
        """Convert an English sentence to a flat list of IPA phoneme tokens."""
        from .phonemes import sentence_to_phonemes

        return sentence_to_phonemes(sentence, self.processor)

    def set_sentence(self, sentence):
        """Set the reference sentence for pronunciation assessment."""
        self.reference_phonemes = self.sentence_to_phonemes(sentence)

    def stream_decode(self, audio_chunks, sample_rate=16000):
        """Evaluate pronunciation from a stream of audio chunks.

        See :func:`openvoice.streaming.stream_decode` for the event format.
        """
        from .streaming import stream_decode

        if self.reference_phonemes is None:
            from .exceptions import OpenVoiceError
            raise OpenVoiceError("Reference sentence not set; call set_sentence() first or pass target sentence into OpenVoiceModel constructor.")
        return stream_decode(audio_chunks, self.reference_phonemes, self.processor,
                             self.model, self.device, sample_rate)
