"""Model loading helpers and the high-level PronounceAssessModel."""

DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"


def load_model(device, model_name=DEFAULT_MODEL, hf_token=None):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    if hf_token is not None:
        model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=hf_token).eval()
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
    model.to(device)
    return processor, model


class PronounceAssessModel:
    def __init__(self, sentence=None, model_name=DEFAULT_MODEL, device=None, hf_token=None):
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        if hf_token is not None:
            self.hf_token = hf_token
        else:
            self.hf_token = None
            print("Hint: you can pass a HuggingFace token to PronounceAssessModel() to avoid rate limits when loading the model.")
        from .utils.prosody_eval import warmup_async
        warmup_async()  # JIT-compile pyin in the background while the model loads
        self.processor, self.model = load_model(device, model_name, self.hf_token)
        self.reference_phonemes = None
        if sentence is not None:
            self.set_sentence(sentence)

    def sentence_to_phonemes(self, sentence):
        from .phonemes import sentence_to_phonemes

        return sentence_to_phonemes(sentence, self.processor)

    def get_phonemes(self):
        from .phonemes import vocabulary

        return vocabulary(self.processor)

    def set_sentence(self, sentence):
        self.reference_phonemes = self.sentence_to_phonemes(sentence)

    def stream_decode(self, audio_chunks, sample_rate=16000):
        from .streaming import stream_decode

        if self.reference_phonemes is None:
            from .exceptions import PronounceAssessError
            raise PronounceAssessError("Reference sentence not set; call set_sentence() first or pass target sentence into PronounceAssessModel constructor.")
        return stream_decode(audio_chunks, self.reference_phonemes, self.processor,
                             self.model, self.device, sample_rate)
