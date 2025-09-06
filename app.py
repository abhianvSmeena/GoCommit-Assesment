# app.py
"""
Interactive CLI: press-to-talk mic, file, or text input.
Shows sentiment & latency. Optional retrieval hits debug table (SHOW_HITS_DEBUG env).
"""
import os
import time
import logging
import tempfile
import threading
import numpy as np

# Windows immediate key detection
try:
    import msvcrt
    _HAS_MS = True
except Exception:
    msvcrt = None
    _HAS_MS = False

import sounddevice as sd
import soundfile as sf

from AIVoiceAssistant import AIVoiceAssistant
import voice_service as vs
import config

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# faster_whisper optional
try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

def _wait_for_enter():
    if _HAS_MS:
        while True:
            ch = msvcrt.getwch()
            if ch == '\r' or ch == '\n':
                return
    else:
        input()  # fallback

def record_press_to_stop(out_path: str, sample_rate: int = 16000, channels: int = 1):
    frames = []
    def callback(indata, frames_count, time_info, status):
        if status:
            logger.debug("Sounddevice status: %s", status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype='float32', callback=callback)
    stream.start()
    start_ts = time.time()
    print("Recording... Press Enter to STOP.")
    stop_thread = threading.Event()

    def timer_worker():
        while not stop_thread.is_set():
            elapsed = time.time() - start_ts
            print(f"\rRecording: {elapsed:.1f}s (press Enter to stop) ", end="", flush=True)
            time.sleep(0.2)
        print()

    t = threading.Thread(target=timer_worker, daemon=True)
    t.start()
    try:
        _wait_for_enter()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_thread.set()
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    if not frames:
        raise RuntimeError("No audio captured.")
    audio_np = np.concatenate(frames, axis=0)
    # convert float32 [-1,1] to int16 PCM
    try:
        pcm16 = (audio_np * 32767).astype(np.int16)
    except Exception:
        pcm16 = audio_np.astype(np.int16)
    sf.write(out_path, pcm16, samplerate=sample_rate, subtype='PCM_16')
    return out_path

def transcribe_with_whisper(model_name: str, audio_path: str):
    if not _HAS_WHISPER:
        raise RuntimeError("faster_whisper not installed.")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    return text

def print_hits_table(hits):
    if not hits:
        return
    print("\n--- Retrieval hits ---")
    for i, h in enumerate(hits, 1):
        md = h.get("metadata", {}) or {}
        src = md.get("source", "unknown")
        page = md.get("page", "")
        print(f"{i:02d}. id={h.get('id')} src={src} page={page} vec_sim={h.get('vec_sim',0):.4f} tfidf={h.get('tfidf_sim',0):.4f} comb={h.get('combined_score',0):.4f}")

def main():
    print("Starting Voice-RAG Assistant (press Enter to start/stop recording). Ctrl+C to exit.")
    assistant = AIVoiceAssistant()
    show_hits_debug = os.getenv("SHOW_HITS_DEBUG", "false").lower() in ("1","true","yes")

    while True:
        try:
            print("\nModes: m=mic, f=file, t=text, q=quit")
            mode = input("Select mode [m/f/t/q]: ").strip().lower()
            if mode == "q":
                print("Exiting.")
                break

            transcription = None

            if mode == "m":
                tmp_wav = os.path.join(tempfile.gettempdir(), f"voice_rag_mic_{int(time.time())}.wav")
                print("Press Enter to START recording.")
                _wait_for_enter()
                try:
                    record_press_to_stop(tmp_wav, sample_rate=16000, channels=1)
                except Exception as e:
                    print("Recording failed:", e)
                    try: os.remove(tmp_wav)
                    except: pass
                    continue

                if _HAS_WHISPER:
                    try:
                        transcription = transcribe_with_whisper(config.WHISPER_MODEL, tmp_wav)
                    except Exception as e:
                        print("Transcription failed:", e)
                        transcription = None
                else:
                    print("Install faster-whisper to transcribe live audio: pip install faster-whisper")
                    transcription = None

                try: os.remove(tmp_wav)
                except: pass

                if not transcription:
                    continue

            elif mode == "f":
                path = input("Enter audio file path: ").strip()
                if not path or not os.path.exists(path):
                    print("File not found.")
                    continue
                if _HAS_WHISPER:
                    try:
                        transcription = transcribe_with_whisper(config.WHISPER_MODEL, path)
                    except Exception as e:
                        print("Transcription failed:", e)
                        continue
                else:
                    print("Install faster-whisper to transcribe files: pip install faster-whisper")
                    continue

            elif mode == "t":
                transcription = input("Type your query: ").strip()
                if not transcription:
                    continue
            else:
                print("Unknown mode.")
                continue

            print("\nUser (transcription):", transcription)

            # call assistant and measure wall-clock
            start_all = time.time()
            result = assistant.interact_with_llm(transcription)
            total_wall = time.time() - start_all

            answer = result.get("answer", "")
            sentiment = result.get("sentiment", {"label":"NEUTRAL","score":0.0})
            latency_inner = result.get("latency", None)
            hits = result.get("hits", [])

            print("\n--- Assistant Answer ---\n")
            print(answer)
            print("\nSources:")
            for s in result.get("sources", []):
                print("-", s)

            print("\nSentiment:", sentiment)
            if latency_inner is not None:
                print(f"LLM latency (assistant measured): {latency_inner:.2f}s")
            print(f"Total wall-clock (transcription+retrieval+generation): {total_wall:.2f}s")

            if show_hits_debug:
                print_hits_table(hits)
            import re 

            # Clean answer for TTS (strip markdown, short bracketed SOURCE tags)
            clean = re.sub(r"[*_`#>\[\]]", "", answer)
            vs.play_text_to_speech(clean, background=True)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            logger.exception("Main loop error: %s", e)
            time.sleep(1)

if __name__ == "__main__":
    main()

