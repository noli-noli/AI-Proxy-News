# speech2text/api_client/speech2text_client.py
# ===============================================================
# Google Cloud Speech-to-Text v2  ストリーミングクライアント
# ・マイク（pyaudio）→ API → 転写結果をリアルタイム表示／保存
# ・ローカル音声ファイルのストリーミング転写もサポート
# ・設定値／認証情報は .env で管理
# ===============================================================

import os
import queue
import threading
from typing import Iterable, List, Callable, Optional

import pyaudio             # マイク入力
from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as speech_types


class Speech2TextClient:
    """Google Cloud Speech-to-Text v2 用クライアント"""

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def __init__(
        self,
        language_codes: List[str] = ("ja-JP",),   # 多言語なら複数与える
        sample_rate: int = 16_000,
        model: str = "chirp",                     # long / chirp など
        enable_auto_decoding: bool = True,
    ) -> None:
        load_dotenv()  # .env を読み込む

        # 必須環境変数
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT が .env に設定されていません。")

        # 認証は GOOGLE_APPLICATION_CREDENTIALS に依存（API キーも可）
        self.client = SpeechClient()
        self.recognizer_path = (
            f"projects/{self.project_id}/locations/global/recognizers/_"
        )

        self.language_codes = list(language_codes)
        self.sample_rate = sample_rate
        self.model = model
        self.enable_auto_decoding = enable_auto_decoding

    # ------------------------------------------------------------------
    # ストリーミング用リクエストジェネレータ
    # ------------------------------------------------------------------
    def _request_stream(
        self, audio_generator: Iterable[bytes]
    ) -> Iterable[speech_types.StreamingRecognizeRequest]:
        """
        1. 最初にストリーミング設定を送信
        2. 続いて audio チャンクを順次送信
        """
        if self.enable_auto_decoding:
            decoding_cfg = speech_types.AutoDetectDecodingConfig()
            recognition_cfg = speech_types.RecognitionConfig(
                auto_decoding_config=decoding_cfg,
                language_codes=self.language_codes,
                model=self.model,
            )
        else:
            decoding_cfg = speech_types.DecodingConfig(
                encoding=speech_types.DecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
            )
            recognition_cfg = speech_types.RecognitionConfig(
                decoding_config=decoding_cfg,
                language_codes=self.language_codes,
                model=self.model,
            )

        streaming_cfg = speech_types.StreamingRecognitionConfig(
            config=recognition_cfg
        )

        # ① 設定リクエスト
        yield speech_types.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=streaming_cfg,
        )

        # ② オーディオストリーム
        for chunk in audio_generator:
            yield speech_types.StreamingRecognizeRequest(audio=chunk)

    # ------------------------------------------------------------------
    # マイク入力のリアルタイム文字起こし
    # ------------------------------------------------------------------
    def stream_microphone(
        self,
        save_transcript: bool = True,
        save_path: str = "transcript.txt",
        interim_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        マイク音声をリアルタイム転写し、結果を任意で保存する。

        Args:
            save_transcript: True なら最終転写結果をファイルへ追記
            save_path: 保存先パス
            interim_callback: interim / final 結果を受け取る関数
                              (引数: transcript str)。指定しなければ標準出力に表示。
        """
        CHUNK = int(self.sample_rate / 10)  # 100 ms
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        audio_interface = pyaudio.PyAudio()
        audio_stream = audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=CHUNK,
        )

        q: queue.Queue[bytes] = queue.Queue()
        stop_event = threading.Event()

        # --- マイク取り込みスレッド --------------------------------------
        def _capture():
            while not stop_event.is_set():
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                q.put(data)

        threading.Thread(target=_capture, daemon=True).start()

        # --- gRPC に渡すジェネレータ ------------------------------------
        def _audio_gen():
            while not stop_event.is_set():
                chunk = q.get()
                if chunk is None:
                    return
                yield chunk

        # --- ストリーミング認識 -----------------------------------------
        try:
            responses = self.client.streaming_recognize(
                requests=self._request_stream(_audio_gen())
            )
            for resp in responses:
                for result in resp.results:
                    transcript = result.alternatives[0].transcript
                    # interim 表示
                    if interim_callback:
                        interim_callback(transcript)
                    else:
                        print(f"\r{transcript}", end="", flush=True)

                    # final 結果なら保存
                    if result.is_final and save_transcript:
                        with open(save_path, "a", encoding="utf-8") as f:
                            f.write(transcript + "\n")

        finally:
            stop_event.set()
            q.put(None)
            audio_stream.stop_stream()
            audio_stream.close()
            audio_interface.terminate()

    # ------------------------------------------------------------------
    # ローカル音声ファイルのストリーミング転写
    # ------------------------------------------------------------------
    def transcribe_file(self, filepath: str) -> List[str]:
        """
        大きなファイルでも gRPC ストリームで送り込んで転写。

        Returns:
            最終転写結果（文単位）のリスト
        """
        with open(filepath, "rb") as f:
            data = f.read()

        # 25 KB/メッセージ制限を考慮して分割
        CHUNK = 25_000
        chunks = [data[i : i + CHUNK] for i in range(0, len(data), CHUNK)]

        responses = self.client.streaming_recognize(
            requests=self._request_stream(chunks)
        )

        final_transcripts: List[str] = []
        for resp in responses:
            for result in resp.results:
                if result.is_final:
                    final_transcripts.append(result.alternatives[0].transcript)
        return final_transcripts
