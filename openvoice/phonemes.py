"""Phoneme utilities: normalization and text → IPA reference sequences."""

# Pairs of phonemes treated as equivalent during matching (source → target).
# ɔ→ɑ: the TIMIT vocab has no ɔ (cot–caught merger); ʒ→ʃ: voicing pair,
# like v→f, and the vocab has no ʒ.
_NORM_MAP = [("ɹ", "r"), ("ʌ", "ə"), ("v", "f"), ("ɔ", "ɑ"), ("ʒ", "ʃ")]


def normalize(phoneme):
    """Collapse near-equivalent IPA symbols so matching is tolerant."""
    for src, dst in _NORM_MAP:
        phoneme = phoneme.replace(src, dst)
    return phoneme


def vocabulary(processor):
    """Return every phoneme token the model's tokenizer can emit.

    Special tokens (padding, unknown, word delimiter) are excluded, so the
    result is the set of IPA phoneme tokens that can appear in parses,
    ordered by vocabulary id.
    """
    tokenizer = processor.tokenizer
    special = set(tokenizer.all_special_tokens)
    special.add(getattr(tokenizer, "word_delimiter_token", None))
    return [token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
            if token not in special]


def sentence_to_phonemes(sentence, processor):
    """Convert an English sentence to a flat list of IPA phoneme tokens.

    Uses eng_to_ipa for grapheme-to-phoneme conversion, then the model's
    tokenizer to split the IPA string into the tokens the model knows.
    """
    import eng_to_ipa as ipa

    # Hyphenated compounds ("red-faced") miss the dictionary as a unit but
    # resolve fine as separate words.
    words = ipa.convert(sentence.replace("-", " "), keep_punct=False,
                        stress_marks=None).split()
    # eng_to_ipa passes unknown words through as raw spelling plus a '*'
    # marker. Raw letters can never be matched by the acoustic model, and a
    # run of them longer than the aligner's lookahead stalls it permanently,
    # so drop unknown words entirely rather than poison the reference.
    ipa_string = "".join(w for w in words if not w.endswith("*"))
    # Safety net: keep only tokens the model can emit some realization of.
    # Compare normalized forms — references use 'r' where the vocab has 'ɹ'.
    special = set(processor.tokenizer.all_special_tokens)
    norm_vocab = {normalize(t) for t in processor.tokenizer.get_vocab()
                  if t not in special and t not in ("|", " ")}
    return [p for p in processor.tokenizer.tokenize(ipa_string)
            if p not in special and p not in ("|", " ", "ˈ", "ˌ")
            and normalize(p) in norm_vocab]
