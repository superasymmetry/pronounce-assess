"""A voice pronunciation assessment library for Python."""

__version__ = "0.1.0"

from .exceptions import AssessmentError, AudioError, PronounceAssessError
from .models import DEFAULT_MODEL, PronounceAssessModel, load_model
from .phonemes import normalize, sentence_to_phonemes, vocabulary
from .streaming import gop_score, stream_decode

__all__ = [
    "__version__",
    "AssessmentError",
    "AudioError",
    "PronounceAssessError",
    "DEFAULT_MODEL",
    "PronounceAssessModel",
    "gop_score",
    "load_model",
    "normalize",
    "sentence_to_phonemes",
    "stream_decode",
    "vocabulary",
]
