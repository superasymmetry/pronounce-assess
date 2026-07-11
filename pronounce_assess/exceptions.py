"""Exceptions raised by pronounce-assess."""


class PronounceAssessError(Exception):
    """Base class for all pronounce-assess errors."""


class AudioError(PronounceAssessError):
    """Raised when audio cannot be loaded or processed."""


class AssessmentError(PronounceAssessError):
    """Raised when a pronunciation assessment fails."""
