import os
print("✅ local_llm.py started (transformers version)")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use an instruction-tuned model that works well for Q&A and reasoning
MODEL_NAME = "google/flan-t5-base"

_tokenizer = None
_model = None


def load_llm():
    """
    Lazy-load the model & tokenizer only once.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    print(f"🚀 Loading local transformer model: {MODEL_NAME} (this may take a while the first time)...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("✅ Model and tokenizer loaded successfully")
    return _tokenizer, _model


def generate_answer(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Generate an answer using a local transformer model (no API key).
    """
    tokenizer, model = load_llm()

    # You can engineer the prompt a bit to make it behave more like ChatGPT
    full_prompt = (
        "You are a helpful, concise AI assistant. "
        "Answer the user’s question clearly.\n\n"
        f"User: {prompt}\nAssistant:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    print("🎯 Running local_llm.py as script (transformers)")

    try:
        reply = generate_answer("Hello! Who are you?")
        print("\n📝 Model reply:\n")
        print(reply)
    except Exception as e:
        print("\n❌ Error while generating answer:", repr(e))
