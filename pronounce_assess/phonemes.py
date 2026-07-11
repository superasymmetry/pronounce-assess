# Pairs of phonemes treated as equivalent during matching (source → target).
# ɔ→ɑ: the TIMIT vocab has no ɔ (cot–caught merger); ʒ→ʃ: voicing pair,
# like v→f, and the vocab has no ʒ.
_NORM_MAP = [("ɹ", "r"), ("ʌ", "ə"), ("v", "f"), ("ɔ", "ɑ"), ("ʒ", "ʃ")]


def normalize(phoneme):
    for src, dst in _NORM_MAP:
        phoneme = phoneme.replace(src, dst)
    return phoneme


def vocabulary(processor):
    tokenizer = processor.tokenizer
    special = set(tokenizer.all_special_tokens)
    special.add(getattr(tokenizer, "word_delimiter_token", None))
    return [token for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
            if token not in special]


def sentence_to_phonemes(sentence, processor):
    import eng_to_ipa as ipa

    words = ipa.convert(sentence.replace("-", " "), keep_punct=False,
                        stress_marks=None).split()
    ipa_string = "".join(w for w in words if not w.endswith("*"))
    special = set(processor.tokenizer.all_special_tokens)
    norm_vocab = {normalize(t) for t in processor.tokenizer.get_vocab()
                  if t not in special and t not in ("|", " ")}
    return [p for p in processor.tokenizer.tokenize(ipa_string)
            if p not in special and p not in ("|", " ", "ˈ", "ˌ")
            and normalize(p) in norm_vocab]
