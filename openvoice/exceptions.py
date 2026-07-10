"""Exceptions raised by OpenVoice."""


class OpenVoiceError(Exception):
    """Base class for all OpenVoice errors."""


class AudioError(OpenVoiceError):
    """Raised when audio cannot be loaded or processed."""


class AssessmentError(OpenVoiceError):
    """Raised when a pronunciation assessment fails."""
