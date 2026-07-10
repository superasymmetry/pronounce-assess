"""Phoneme utilities: normalization and text → IPA reference sequences."""

# Pairs of phonemes treated as equivalent during matching (source → target).
_NORM_MAP = [("ɹ", "r"), ("ʌ", "ə"), ("v", "f")]


def normalize(phoneme):
    """Collapse near-equivalent IPA symbols so matching is tolerant."""
    for src, dst in _NORM_MAP:
        phoneme = phoneme.replace(src, dst)
    return phoneme


def sentence_to_phonemes(sentence, processor):
    """Convert an English sentence to a flat list of IPA phoneme tokens.

    Uses eng_to_ipa for grapheme-to-phoneme conversion, then the model's
    tokenizer to split the IPA string into the tokens the model knows.
    """
    import eng_to_ipa as ipa

    ipa_string = ipa.convert(sentence).replace(" ", "")
    return [p for p in processor.tokenizer.tokenize(ipa_string) if p != "ˈ"]
