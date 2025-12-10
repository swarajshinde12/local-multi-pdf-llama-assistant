import os
print("✅ local_llm_gguf.py started")

try:
    from llama_cpp import Llama
    print("✅ Imported llama_cpp successfully")
except Exception as e:
    print("❌ Error importing llama_cpp:", repr(e))
    raise

# Build path to models/llm.gguf
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "llm.gguf")

print("🔍 Project root:", PROJECT_ROOT)
print("🔍 Model path:", MODEL_PATH)
print("🔍 Model exists?", os.path.exists(MODEL_PATH))


# Load model once (global)
_llm = None

def load_llm():
    global _llm
    if _llm is not None:
        return _llm

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"GGUF model not found at {MODEL_PATH}")

    print("🚀 Loading GGUF model with llama-cpp...")
    _llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,         # CPU only used a little
    n_gpu_layers=-1,     # USE GPU FOR ALL LAYERS 🔥
    use_mmap=False,      # faster on Windows GPU
    use_mlock=False,
)

    print("✅ GGUF model loaded successfully")
    return _llm


def generate_answer(prompt: str, max_tokens: int = 256) -> str:
    """
    Use local GGUF LLaMA model to answer a prompt.
    """
    llm = load_llm()
    print("🤖 Generating answer from GGUF model...")
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    text = resp["choices"][0]["message"]["content"]
    print("✅ Got response from GGUF model")
    return text


if __name__ == "__main__":
    try:
        reply = generate_answer("Hello! Who are you?")
        print("\n📝 GGUF model reply:\n")
        print(reply)
    except Exception as e:
        print("\n❌ Error while using GGUF model:", repr(e))
