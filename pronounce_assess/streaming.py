from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .phonemes import normalize
from .utils.prosody_eval import evaluate_prosody, warmup_async


@dataclass
class StreamEvent:
    phoneme: Optional[str] = None
    position: Optional[int] = None
    label: str = ""                 # 'correct' | 'mispronounced' | 'omitted' | 'prosody'
    decoded: Optional[str] = None   # what the model actually decoded ('' for omitted)
    target_logit: float = float("nan")
    decoded_logit: float = float("nan")
    gop: Optional[float] = None     # graded Goodness of Pronunciation in (0, 1]; 0.0 omitted
    monotony_score: Optional[float] = None
    rhythm_score: Optional[float] = None
    boundary_score: Optional[float] = None
    speaking_rate: Optional[float] = None


def gop_score(target_logit, decoded_logit):
    if np.isnan(target_logit) or np.isnan(decoded_logit):
        return None
    return float(np.exp(min(target_logit - decoded_logit, 0.0)))


def stream_decode(audio_chunks, reference_phonemes, processor, model,
                  device, sample_rate=16000, text=""):
    """Evaluate pronunciation from a stream of audio chunks.

    Args:
        audio_chunks: iterable of float32 numpy arrays at sample_rate Hz.
                      The caller signals end-of-stream by exhausting the iterable.
        reference_phonemes: flat list of IPA phoneme strings to match against, in order.
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC
        device: torch device string
        sample_rate: audio sample rate in Hz (default 16000)
        text: reference sentence text, used for the prosody boundary-tone check.

    Yields:
        StreamEvent for each matched reference phoneme (prosody fields None),
        then one final StreamEvent with label 'prosody' carrying the scores
        from evaluate_prosody once the stream is exhausted.
    """
    

    reference_phonemes = [p for p in reference_phonemes if p != "ˈ"]
    pointer = 0
    phoneme2id = processor.tokenizer.get_vocab()
    special_ids = set(processor.tokenizer.all_special_ids)
    norm2ids = {}
    for tok, tid in phoneme2id.items():
        if tid not in special_ids and tok not in ("|", " "):
            norm2ids.setdefault(normalize(tok), []).append(tid)
    logit_threshold = 5.0
    lookahead = 3

    all_audio = []
    warmup_async()  # JIT-compile pyin in the background while audio streams in

    for i, chunk in enumerate(audio_chunks):
        all_audio.append(chunk)
        if pointer >= len(reference_phonemes):
            continue  # keep draining the stream so prosody sees the full utterance

        inputs = processor(chunk, sampling_rate=sample_rate,
                           return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
        logits_np = logits[0].cpu().numpy()
        blank_id = processor.tokenizer.pad_token_id
        collapsed = []
        collapsed_frames = []
        prev = None
        for fi, idx in enumerate(predicted_ids):
            if idx != prev:
                if idx != blank_id:
                    collapsed.append(idx)
                    collapsed_frames.append(fi)
                prev = idx

        special = set(processor.tokenizer.all_special_ids)
        tokens = processor.tokenizer.convert_ids_to_tokens(collapsed)
        chunk_phonemes = []
        phoneme_logits = []
        chunk_frames = []
        for fi, idx, t in zip(collapsed_frames, collapsed, tokens):
            if processor.tokenizer.convert_tokens_to_ids(t) not in special and t not in ("|", " "):
                chunk_phonemes.append(normalize(t))
                phoneme_logits.append(logits_np[fi, idx])
                chunk_frames.append(fi)

        if chunk_phonemes and pointer < len(reference_phonemes):
            current_count = sum(
                1 for di, dp in enumerate(chunk_phonemes)
                if pointer + di < len(reference_phonemes)
                and dp == normalize(reference_phonemes[pointer + di])
            )
            if current_count == 0:
                best_count, best_skip = 0, 0
                back = min(lookahead, pointer)
                fwd  = min(lookahead * 2 + 1, len(reference_phonemes) - pointer)
                for skip in list(range(-back, 0)) + list(range(1, fwd)):
                    count = sum(
                        1 for di, dp in enumerate(chunk_phonemes)
                        if 0 <= pointer + skip + di < len(reference_phonemes)
                        and dp == normalize(reference_phonemes[pointer + skip + di])
                    )
                    if count > best_count:
                        best_count, best_skip = count, skip
                if best_count >= 2:
                    if best_skip > 0:
                        for k in range(best_skip):
                            yield StreamEvent(
                                phoneme=reference_phonemes[pointer + k],
                                position=pointer + k,
                                label="omitted",
                                decoded="",
                                gop=0.0,
                            )
                    pointer += best_skip

        for j, (fi, p, lv) in enumerate(zip(chunk_frames, chunk_phonemes, phoneme_logits)):
            if pointer >= len(reference_phonemes):
                continue
            target_p = normalize(reference_phonemes[pointer])
            target_ids = norm2ids.get(target_p, [])
            target_lv = max((float(logits_np[fi, t]) for t in target_ids),
                            default=float("nan"))
            top3_ids = set(logits_np[fi].argsort()[-3:].tolist())
            target_in_top3 = any(t in top3_ids for t in target_ids)
            if p == target_p or (target_ids and target_lv > logit_threshold) or target_in_top3:
                label = "correct"
                pos = pointer
                pointer += 1
            elif target_ids and target_lv > 0:
                next_is_target = j + 1 < len(chunk_phonemes) and chunk_phonemes[j + 1] == target_p
                if next_is_target:
                    label = "insertion"
                    pos = None
                else:
                    label = "mispronounced"
                    pos = pointer
                    pointer += 1
            else:
                # Target logit is ≤ 0 or unknown — check if the reference phoneme was
                # simply omitted and the decoded phoneme matches something ahead.
                skip_to = next(
                    (k for k in range(1, min(lookahead + 1, len(reference_phonemes) - pointer))
                     if p == normalize(reference_phonemes[pointer + k])),
                    None,
                )
                if skip_to is not None:
                    for k in range(skip_to):
                        yield StreamEvent(
                            phoneme=reference_phonemes[pointer + k],
                            position=pointer + k,
                            label="omitted",
                            decoded="",
                            gop=0.0,
                        )
                    pointer += skip_to
                    label = "correct"
                    pos = pointer
                    # target_lv was read for the phoneme we just marked
                    # omitted; re-read it for the advanced pointer so the
                    # yielded event scores the phoneme it actually matched.
                    target_ids = norm2ids.get(normalize(reference_phonemes[pointer]), [])
                    target_lv = max((float(logits_np[fi, t]) for t in target_ids),
                                    default=float("nan"))
                    pointer += 1
                else:
                    label = "insertion"
                    pos = None
            if pos is not None:
                yield StreamEvent(
                    phoneme=reference_phonemes[pos],
                    position=pos,
                    label=label,
                    decoded=p,
                    target_logit=target_lv,
                    decoded_logit=lv,
                    gop=gop_score(target_lv, lv),
                )

    if all_audio:
        yield StreamEvent(label="prosody",
                          **evaluate_prosody(np.concatenate(all_audio), text, sr=sample_rate))