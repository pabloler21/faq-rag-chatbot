"""Diagnostico de LM Studio. Correr antes de construir nada encima."""
import numpy as np
from openai import OpenAI

BASE_URL = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-qwen3-embedding-0.6b"
LLM_MODEL = "qwen/qwen3.5-4b"

client = OpenAI(base_url=BASE_URL, api_key="lm-studio")


def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    vector = np.array(response.data[0].embedding, dtype="float32")
    return vector / np.linalg.norm(vector)


def main() -> None:
    a = embed("How do I cancel my subscription?")
    b = embed("What is the process to terminate my plan?")
    c = embed("The mitochondria is the powerhouse of the cell.")

    print(f"dimension          : {a.shape[0]}")
    print(f"similar   (esperado > 0.6) : {float(a @ b):.4f}")
    print(f"inconexo  (esperado < 0.4) : {float(a @ c):.4f}")

    reply = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(f"llm                : {reply.choices[0].message.content!r}")


if __name__ == "__main__":
    main()
