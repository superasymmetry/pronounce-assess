"""Live microphone demo: read a sentence aloud and see per-phoneme feedback.
"""
from openvoice import OpenVoiceModel
from openvoice.audio import ChunkRecord


def main():
    duration = 5
    chunk_ms = 500
    sample_rate = 16000

    print("Loading model...", flush=True)
    assessor = OpenVoiceModel()
    print("Model loaded.", flush=True)

    reference_sentence = "The quick brown fox jumps over the lazy dog"
    assessor.set_sentence(reference_sentence)

    print(f"Recording for {duration}s, decoding every {chunk_ms}ms...", flush=True)
    with ChunkRecord(duration=duration, chunk_len=int(sample_rate * chunk_ms / 1000), sample_rate=sample_rate) as chunk_record:
        for event in assessor.stream_decode(chunk_record, sample_rate):
            print(f"  -> {event}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
