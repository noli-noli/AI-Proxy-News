from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transcriber import Transcriber  

BASE_DIR = Path(__file__).resolve().parents[2]          # project-root
HTML_DIR = BASE_DIR / "ui" / "html"
INDEX    = HTML_DIR / "ai-proxy-news.html"

app = FastAPI()
# 同一オリジンで動くため CORS は不要だが、開発時の利便で * 許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- STT エンドポイント ----------
tr = Transcriber()

@app.post("/start")
def start_rec():
    if tr.start():
        return {"ok": True}
    return JSONResponse({"ok": False, "msg": "already running"}, status_code=400)

@app.post("/stop")
def stop_rec():
    tr.stop(); return {"ok": True}

@app.get("/status")
def status():
    return tr.status()

@app.get("/pull")
def pull():
    return {"lines": tr.fetch_transcript()}

# ---------- フロントエンド配信 ----------
# 静的ファイル (HTML / JS / CSS) をそのまま配信
app.mount("/", StaticFiles(directory=HTML_DIR, html=True), name="static")

# optional: ルート / だけを INDEX に強制する場合は下記でも可
# @app.get("/")
# def root(): return FileResponse(INDEX)
