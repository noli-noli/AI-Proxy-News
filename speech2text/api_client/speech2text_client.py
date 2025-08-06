"""
stream_recognize_env.py
────────────────────────────────────────────────────────────────
Google Cloud Speech-to-Text でリアルタイム文字起こし

機能
  • コンソールに転写結果を即時表示
  • 経過秒を常時ステータス表示
  • gRPC ストリーム 5 分上限 → 290 秒ごとに自動再接続
  • 無音 5 分で強制終了
  • 終了時に .txt へ保存（ファイル名 = 開始日時）
    ↳ Ctrl-C 直後の interim 文も漏れなく保存

依存
  pip install google-cloud-speech python-dotenv pyaudio
  .env に GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
"""

import os
import re
import sys
import time
import queue
import argparse
import threading
from typing import List

from dotenv import load_dotenv
from google.cloud import speech
from google.api_core import exceptions
import pyaudio

# ──────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────
load_dotenv()

RATE                 = 16_000
CHUNK                = int(RATE / 10)       # 100 ms
DEFAULT_MODEL        = "latest_long"
DEFAULT_LANG         = "ja-JP"
TEXT_SAVE_DIR        = "../save-texts/"
STREAMING_LIMIT_SEC  = 290                  # gRPC 305 s − 余裕
SILENCE_TIMEOUT_SEC  = 300                  # 無音 5 min で終了

os.makedirs(TEXT_SAVE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# マイクストリーム
# ──────────────────────────────────────────────────────────────
class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self.rate, self.chunk = rate, chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream    = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, *_):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *_):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

# ──────────────────────────────────────────────────────────────
# 経過秒表示スレッド
# ──────────────────────────────────────────────────────────────
def display_elapsed(start: float, stop_evt: threading.Event):
    last = -1
    while not stop_evt.is_set():
        sec = int(time.monotonic() - start)
        if sec != last:
            sys.stderr.write(f"\r[elapsed {sec:>6} s] ")
            sys.stderr.flush()
            last = sec
        time.sleep(0.2)

# ──────────────────────────────────────────────────────────────
# ストリーミング結果処理
# ──────────────────────────────────────────────────────────────
def listen_print_loop(
    responses,
    session_start: float,
    last_speech: List[float],
    outfile,
    interim_holder: List[str],
):
    """
    responses         : gRPC ストリーム
    session_start     : セッション開始時間 (monotonic)
    last_speech[0]    : 最後に音声検出した時間を共有
    outfile           : open(...) されたファイルハンドル
    interim_holder[0] : 直近の interim 文を保持
    """
    num_chars_printed = 0
    for resp in responses:
        # セッション時間超過したらリターン → 上位で再接続
        if time.monotonic() - session_start > STREAMING_LIMIT_SEC:
            return

        if not resp.results:
            continue
        result = resp.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite  = " " * max(0, num_chars_printed - len(transcript))
        last_speech[0]      = time.monotonic()
        interim_holder[0]   = transcript  # interim / final 共通で保持

        if not result.is_final:
            # interim 表示
            sys.stdout.write(transcript + overwrite + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            # final 表示 & 保存
            print(transcript + overwrite)
            outfile.write(transcript + "\n")
            outfile.flush()
            interim_holder[0] = ""         # final で確定したのでクリア
            num_chars_printed  = 0

# ──────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text streaming")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lang",  default=DEFAULT_LANG)
    args = parser.parse_args()

    client = speech.SpeechClient()
    recogn_cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=args.lang,
        model=args.model,
        enable_automatic_punctuation=True,
    )
    stream_cfg = speech.StreamingRecognitionConfig(
        config=recogn_cfg,
        interim_results=True,
    )

    # 保存ファイルを先に確保（開始日時で命名）
    stamp     = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tmp_path  = os.path.join(TEXT_SAVE_DIR, f"{stamp}.txt")
    outfile   = open(tmp_path, "a", encoding="utf-8")

    prog_start_wall = time.time()
    prog_start_mono = time.monotonic()
    last_speech     = [prog_start_mono]     # 無音判定用（共有参照）
    interim_buf     = [""]                  # 現在の interim を保持

    # 経過秒スレッド起動
    stop_evt = threading.Event()
    threading.Thread(
        target=display_elapsed,
        args=(prog_start_mono, stop_evt),
        daemon=True,
    ).start()

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            while True:  # Ctrl-C までループ
                sess_start = time.monotonic()

                requests = (
                    speech.StreamingRecognizeRequest(audio_content=c)
                    for c in stream.generator()
                )
                responses = client.streaming_recognize(stream_cfg, requests)

                try:
                    listen_print_loop(
                        responses,
                        sess_start,
                        last_speech,
                        outfile,
                        interim_buf,
                    )
                except exceptions.OutOfRange:
                    print("\n[INFO] Stream limit reached – reconnecting…")
                    continue  # 再接続

                # 無音タイムアウト判定
                if time.monotonic() - last_speech[0] > SILENCE_TIMEOUT_SEC:
                    print("\n[INFO] Silent for 5 min – terminating.")
                    break

    except KeyboardInterrupt:
        print("\n[Ctrl-C] Streaming finished.")
    finally:
        # interim が残っていれば保存してから終了
        if interim_buf[0]:
            outfile.write(interim_buf[0] + "\n")
        outfile.close()
        stop_evt.set()

        # 0 バイトなら削除
        if os.path.getsize(tmp_path) == 0:
            os.remove(tmp_path)
            print("[INFO] No speech captured; nothing saved.")
        else:
            duration = int(time.time() - prog_start_wall)
            print(f"[INFO] Transcript saved → {tmp_path}  ({duration}s)")

if __name__ == "__main__":
    main()