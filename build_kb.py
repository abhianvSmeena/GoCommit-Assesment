# scripts/build_kb.py
from AIVoiceAssistant import AIVoiceAssistant

def build_kb():
    assistant = AIVoiceAssistant()
    print("KB ready. Chroma persisted at:", assistant.kb.persist_directory if hasattr(assistant.kb, "persist_directory") else "unknown")

if __name__ == "__main__":
    build_kb()
