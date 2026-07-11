import queue
import threading
import time

import gradio as gr
import numpy as np

TARGET_SR = 16000

LABEL_COLORS = {
    "correct": "#22c55e",
    "mispronounced": "#ef4444",
    "omitted": "#f97316",
    "not reached": "#9ca3af",
}

_model = None


def get_model():
    """Load the model once, on first use."""
    global _model
    if _model is None:
        from pronounce_assess import PronounceAssessModel

        _model = PronounceAssessModel()
    return _model


def enable_mic():
    get_model()
    return gr.Audio(visible=True), gr.Markdown(visible=False)

def to_float_mono_16k(audio):
    sr, samples = audio
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.dtype.kind in "iu":
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32)
    if sr != TARGET_SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
    return samples


def render_highlights(reference, events, pending_label=None):
    return [(p, events[i].label if i in events else pending_label)
            for i, p in enumerate(reference)]


def render_scores(reference, events, prosody):
    gops = [e.gop for e in events.values() if e.gop is not None]
    lines = []
    if gops:
        lines.append(f"**Overall pronunciation score:** {np.mean(gops):.0%} "
                     f"({len(gops)}/{len(reference)} phonemes assessed)")
    else:
        lines.append("No phonemes could be assessed — try speaking louder or closer to the mic.")
    if prosody is not None:
        def fmt(score):
            return f"{score:.2f}" if score is not None else "n/a"

        lines.append("")
        lines.append("| Prosody | Score |")
        lines.append("|---|---|")
        lines.append(f"| Monotony | {fmt(prosody.monotony_score)} |")
        lines.append(f"| Rhythm | {fmt(prosody.rhythm_score)} |")
        lines.append(f"| Boundary tone | {fmt(prosody.boundary_score)} |")
        lines.append(f"| Speaking rate | {fmt(prosody.speaking_rate)} |")
    return "\n".join(lines)


class LiveSession:
    def __init__(self, sentence):
        model = get_model()
        self.sentence = sentence
        self.reference = [p for p in model.sentence_to_phonemes(sentence) if p != "ˈ"]
        self.chunks = queue.Queue()
        self.events = {}
        self.prosody = None
        self.finished = False
        self.last_chunk_at = time.monotonic()
        self.thread = threading.Thread(target=self._decode, args=(model,), daemon=True)
        self.thread.start()

    def add_chunk(self, samples):
        self.last_chunk_at = time.monotonic()
        self.chunks.put(samples)

    def _iter_chunks(self):
        while True:
            chunk = self.chunks.get()
            if chunk is None:
                return
            yield chunk

    def _decode(self, model):
        from pronounce_assess.streaming import stream_decode

        for event in stream_decode(self._iter_chunks(), self.reference,
                                   model.processor, model.model, model.device,
                                   TARGET_SR, text=self.sentence):
            if event.label == "prosody":
                self.prosody = event
            elif event.position is not None:
                self.events[event.position] = event

    def finish(self):
        deadline = time.monotonic() + 4.0
        while (time.monotonic() < deadline
               and time.monotonic() - self.last_chunk_at < 1.5):
            time.sleep(0.1)
        self.finished = True
        self.chunks.put(None)
        self.thread.join(timeout=60)


def start_live(sentence):
    if not sentence or not sentence.strip():
        raise gr.Error("Enter a reference sentence first.")
    return LiveSession(sentence), [], ""


def stream_live(session, sentence, chunk):
    if session is None:  # stream event fired before start_recording finished
        session = LiveSession(sentence)
    if session.finished:  # straggler event after stop — results already rendered
        return session, gr.skip()
    if chunk is not None:
        session.add_chunk(to_float_mono_16k(chunk))
    events = dict(session.events)
    spoken = [(session.reference[i], events[i].label) for i in sorted(events)]
    return session, spoken


def finish_live(session):
    if session is None:
        return None, gr.skip(), gr.skip()
    session.finish()
    highlights = render_highlights(session.reference, session.events,
                                   pending_label="not reached")
    return session, highlights, render_scores(session.reference, session.events,
                                              session.prosody)


def assess(sentence, audio):
    if not sentence or not sentence.strip():
        raise gr.Error("Enter a reference sentence first.")
    if audio is None:
        raise gr.Error("Record or upload some audio first.")

    samples = to_float_mono_16k(audio)
    model = get_model()
    model.set_sentence(sentence)
    reference = [p for p in model.reference_phonemes if p != "ˈ"]

    events = {}
    prosody = None
    for event in model.stream_decode([samples], TARGET_SR):
        if event.label == "prosody":
            prosody = event
        elif event.position is not None:
            events[event.position] = event

    highlights = render_highlights(reference, events, pending_label="not reached")
    return highlights, render_scores(reference, events, prosody)


def build_demo():
    with gr.Blocks(title="pronounce-assess — Pronunciation Assessment") as demo:
        gr.Markdown(
            "# pronounce-assess pronunciation assessment\n"
            "Try out pronounce-assess's real-time pronunciation assessment in a gradio demo."
        )
        sentence = gr.Textbox(
            label="Reference sentence",
            value="The quick brown fox jumps over the lazy dog",
        )
        with gr.Tab("Live (microphone)"):
            loading = gr.Markdown("Loading the pronunciation model…")
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                           label="Speak",
                           visible=False)
            live_state = gr.State(None)
            live_phonemes = gr.HighlightedText(label="Phoneme-level result",
                                               color_map=LABEL_COLORS)
            live_scores = gr.Markdown()
            mic.start_recording(start_live, inputs=[sentence],
                                outputs=[live_state, live_phonemes, live_scores])
            mic.stream(stream_live, inputs=[live_state, sentence, mic],
                       outputs=[live_state, live_phonemes], stream_every=0.7)
            mic.stop_recording(finish_live, inputs=[live_state],
                               outputs=[live_state, live_phonemes, live_scores])
        with gr.Tab("Upload a clip"):
            clip = gr.Audio(sources=["upload", "microphone"], type="numpy",
                            label="Your recording")
            button = gr.Button("Assess pronunciation", variant="primary")
            clip_phonemes = gr.HighlightedText(label="Phoneme-level result",
                                               color_map=LABEL_COLORS)
            clip_scores = gr.Markdown()
            button.click(assess, inputs=[sentence, clip],
                         outputs=[clip_phonemes, clip_scores])
        demo.load(enable_mic, outputs=[mic, loading])
    return demo


if __name__ == "__main__":
    build_demo().launch(share=True)
