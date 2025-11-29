from gtts import gTTS
from playsound import playsound
import tempfile
import os
import threading
import queue
import time

# Queue để xử lý tuần tự, không chồng chéo
_speech_queue = queue.Queue()
_is_speaking = False
_last_speak_time = 0
SPEAK_COOLDOWN = 1.0  # Tối thiểu 1 giây giữa các lần nói

def _speech_worker():
    """Worker thread xử lý queue tuần tự."""
    global _is_speaking
    while True:
        try:
            text, lang = _speech_queue.get(timeout=1)
            _is_speaking = True
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    filename = fp.name
                tts = gTTS(text=text, lang=lang)
                tts.save(filename)
                playsound(filename)
                os.remove(filename)
            except Exception as e:
                print("TTS error:", e)
            finally:
                _is_speaking = False
                _speech_queue.task_done()
        except queue.Empty:
            continue

# Khởi động worker thread
_worker_thread = threading.Thread(target=_speech_worker, daemon=True)
_worker_thread.start()

def speak(text, lang='vi'):
    """Thêm text vào queue, bỏ qua nếu queue đang đầy hoặc cooldown chưa hết."""
    global _last_speak_time
    
    now = time.time()
    
    # Bỏ qua nếu chưa hết cooldown
    if now - _last_speak_time < SPEAK_COOLDOWN:
        return
    
    # Bỏ qua nếu queue đã có nhiều item (tránh lag)
    if _speech_queue.qsize() > 2:
        return
    
    _last_speak_time = now
    _speech_queue.put((text, lang))