# voice_service.py
import os
import tempfile
import threading
import logging
import re

TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")
logger = logging.getLogger("voice_service")
logging.basicConfig(level=logging.INFO)

def _speak_with_pyttsx3(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.warning("pyttsx3 failed: %s — falling back to gTTS", e)
        _speak_with_gtts(text)

def _speak_with_gtts(text: str):
    try:
        from gtts import gTTS
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=text, lang="en")
        tts.save(tmp.name)
        _play_file(tmp.name)
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    except Exception as e:
        logger.exception("gTTS failed: %s", e)

def _play_file(path: str):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
    except Exception as e:
        logger.debug("pygame playback failed: %s", e)
        if os.name == "nt":
            os.system(f'start /min wmplayer "{path}"')
        else:
            os.system(f'ffplay -nodisp -autoexit "{path}" >/dev/null 2>&1 &')

def play_text_to_speech(text: str, background: bool = True):
    if not text:
        return
    # additional cleaning: remove URLs and long bracketed tokens
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[SOURCE\s*\d+:.*?\]", "", text)
    # collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    def _worker(t):
        logger.info("Using TTS engine: %s", TTS_ENGINE)
        if TTS_ENGINE == "pyttsx3":
            _speak_with_pyttsx3(t)
        else:
            _speak_with_gtts(t)
    if background:
        thr = threading.Thread(target=_worker, args=(text,), daemon=True)
        thr.start()
    else:
        _worker(text)

