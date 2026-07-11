from openvoice import OpenVoiceModel
from openvoice.audio import ChunkRecord


def main():
    print("Loading model...", flush=True)
    assessor = OpenVoiceModel()
    assessor.set_sentence("The quick brown fox jumps over the lazy dog.")
    print("Model loaded. Please start speaking.", flush=True)

    with ChunkRecord(duration=5, chunk_len=int(16000 * 500 / 1000), sample_rate=16000) as chunk_record:
        for event in assessor.stream_decode(chunk_record, 16000):
            print(event, flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
