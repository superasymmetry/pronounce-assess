
class PronounceAssessError(Exception):
    pass

class AudioError(PronounceAssessError):
    pass

class AssessmentError(PronounceAssessError):
    pass