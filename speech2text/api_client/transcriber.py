"""
Transcriber
────────────────────────────────────────────────────────────────
Google Cloud Speech-to-Text をスレッドで動かし、
・/start で録音＋転写開始
・/stop で録音停止
・result.is_final が来るたびにブラウザへ push＆同時にファイルへ逐次追記
・無音 5 分で強制停止
依存: google-cloud-speech, pyaudio, python-dotenv
"""

import time, threading, queue, pathlib, os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()                                                # ADC 用

from google.cloud import speech
from google.api_core import exceptions
import pyaudio

# ───── パラメータ ─────────────────────────────────
RATE   = 16_000
CHUNK  = int(RATE / 10)          # 100 ms
LIMIT  = 290                     # 1 セッション 5 min(305s) − ε
SILENT = 300                     # 無音 5 min で停止
SAVE_DIR = pathlib.Path(__file__).resolve().parents[2] / "save-texts"
SAVE_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────────


class Transcriber:
    def __init__(self, lang="ja-JP", model="latest_long") -> None:
        self.lang, self.model = lang, model
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        self._alive  = False

        self._queue  = queue.Queue()        # ブラウザへ返すキュー
        self._file   = None                 # 追記用ファイルハンドル
        self._file_path: Optional[pathlib.Path] = None

    # ---------- public ----------
    def start(self) -> bool:
        with self._lock:
            if self._alive:
                return False
            self._stop.clear()

            # ファイルを用意 (時刻で命名)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self._file_path = SAVE_DIR / f"{stamp}.txt"
            self._file = open(self._file_path, "a", encoding="utf-8")

            # スレッド開始
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._alive = True
            return True

    def stop(self) -> None:
        with self._lock:
            if not self._alive:
                return
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2)
            self._close_file()
            self._alive = False

    def status(self) -> dict:
        return {"running": self._alive, "queue_size": self._queue.qsize()}

    def fetch_transcript(self) -> List[str]:
        items = []
        while not self._queue.empty():
            items.append(self._queue.get())
        return items

    # ---------- internal ----------
    def _close_file(self):
        if self._file:
            self._file.close()
            self._file = None
            print(f"[INFO] transcript saved → {self._file_path}")

    def _run(self):
        try:
            client = speech.SpeechClient()
        except Exception as e:
            self._queue.put(f"[ERROR] {e}")
            self._close_file()
            self._alive = False
            return

        cfg = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=self.lang,
            model=self.model,
            enable_automatic_punctuation=True,
        )
        stream_cfg = speech.StreamingRecognitionConfig(
            config=cfg, interim_results=True
        )

        last_speech = time.monotonic()

        with _MicStream(RATE, CHUNK) as mic:
            while not self._stop.is_set():
                sess_start = time.monotonic()

                requests = (
                    speech.StreamingRecognizeRequest(audio_content=b)
                    for b in mic.generator()
                )

                try:
                    for resp in client.streaming_recognize(stream_cfg, requests):
                        if self._stop.is_set():
                            break
                        if not resp.results:
                            continue
                        result = resp.results[0]
                        if not result.alternatives:
                            continue

                        transcript = result.alternatives[0].transcript
                        last_speech = time.monotonic()

                        # final のみキュー＆ファイルに書く
                        if result.is_final:
                            self._queue.put(transcript)
                            if self._file:
                                self._file.write(transcript + "\n")
                                self._file.flush()

                        # セッション時間超え → ループ脱出して再接続
                        if time.monotonic() - sess_start > LIMIT:
                            break

                    # 無音タイムアウト
                    if time.monotonic() - last_speech > SILENT:
                        self._queue.put("[ERROR] Silent for 5 min – auto stop.")
                        break

                except exceptions.OutOfRange:
                    continue        # gRPC 305 s 超え → 再接続

        # 正常終了 or stop/無音
        self._close_file()
        self._alive = False


class _MicStream:
    """ 内部用マイクラッパ（プログラム終了まで使い回す） """
    def __init__(self, rate, chunk):
        self.rate, self.chunk = rate, chunk
        self.buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill,
        )
        self.closed = False
        return self

    def __exit__(self, *a):
        self.stream.stop_stream(); self.stream.close()
        self.closed = True; self.buff.put(None); self.audio.terminate()

    def _fill(self, in_data, *_):
        self.buff.put(in_data); return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self.buff.get()
            if chunk is None: return
            data = [chunk]
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None: return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)
