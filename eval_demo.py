# eval_demo.py
import time
from AIVoiceAssistant import AIVoiceAssistant

SCENARIOS = [
    "Explain how an insulin pump works to a non-technical person in simple terms.",
    "I've been feeling dizzy since starting steroids; I have stage 2 CKD. Is this normal? I'm worried.",
    "Explain quantum computing like I'm 10, but make it enthusiastic and include a simple metaphor."
]

def run_demo():
    assistant = AIVoiceAssistant()
    for s in SCENARIOS:
        print("\nQUERY:", s)
        start = time.time()
        res = assistant.interact_with_llm(s)
        elapsed = time.time() - start
        print("ANSWER:\n", res.get("answer"))
        print("SOURCES:", res.get("sources"))
        print("SENTIMENT:", res.get("sentiment"))
        print(f"Elapsed: {elapsed:.2f}s, reported latency: {res.get('latency'):.2f}s")

if __name__ == "__main__":
    run_demo()
