"""
stream_recognize_env.py  (モデル/言語切替版)

依存:
  pip install google-cloud-speech python-dotenv pyaudio
  .env に GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
"""

import os, sys, re, queue, argparse
from dotenv import load_dotenv
from google.cloud import speech
import pyaudio

# ---------- 環境変数読み込み ----------
load_dotenv()

# ---------- 定数 ----------
RATE          = 16000
CHUNK         = int(RATE / 10)   # 100 ms
DEFAULT_MODEL = "latest_long"   # latest_short / latest_long / command_and_search ...
DEFAULT_LANG  = "ja-JP"          # ja-JP で日本語
TEXT_SAVE_DIR = "../save-texts/"


if not os.path.exists(TEXT_SAVE_DIR):
    os.makedirs(TEXT_SAVE_DIR)
    
# ---------- マイク入力 ----------
class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate, self._chunk = rate, chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16, channels=1,
            rate=self._rate, input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None: return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None: return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

# ---------- 文字起こし結果表示 ----------
def listen_print_loop(responses):
    outfile = open(f"{TEXT_SAVE_DIR}transcript.txt", "a", encoding="utf-8")  # ★追記
    num_chars_printed = 0
    for response in responses:
        if not response.results: continue
        result = response.results[0]
        if not result.alternatives: continue
        transcript = result.alternatives[0].transcript
        overwrite = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite)
            outfile.write(transcript + "\n")   # ★保存
            outfile.flush()
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting.."); break
            num_chars_printed = 0

# ---------- メイン ----------
def main():
    parser = argparse.ArgumentParser(description="GCP Speech-to-Text streaming demo")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"使用モデル (default: {DEFAULT_MODEL})")
    parser.add_argument("--lang", default=DEFAULT_LANG,
                        help=f"言語コード (default: {DEFAULT_LANG})")
    args = parser.parse_args()

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=args.lang,
        model=args.model,                  # ← 追加
        enable_automatic_punctuation=True  # 好みで
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in audio_gen)
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)

if __name__ == "__main__":
    main()
