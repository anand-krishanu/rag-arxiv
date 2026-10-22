import json
import ollama

MODEL = "mistral:latest"

SYSTEM = (
    "You answer questions using ONLY the numbered sources.\n"
    "Rules:\n"
    "1. Answer only if a source states the exact thing asked, for the exact entity named in the question. "
    "A similar entity does not count (for example GPT-4o mini is not GPT-4o).\n"
    "2. Copy numbers exactly as they appear in the source.\n"
    "3. If no source states it, set found to false, answer to an empty string, and sources to an empty list.\n"
    "4. Never use your own knowledge. A true fact that is not in the sources is still not an answer.\n"
    "5. Write one or two sentences. Never answer with only a source number."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "sources"],
}


def build_prompt(question, chunks):
    sources = []
    for i, c in enumerate(chunks, 1):
        sources.append(f"[{i}] (paper {c['paper_id']}, page {c['page'] + 1})\n{c['text']}")
    return "Sources:\n\n" + "\n\n".join(sources) + f"\n\nQuestion: {question}"


def answer(question, chunks):
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(question, chunks)}],
        format=SCHEMA,
        options={"temperature": 0, "num_ctx": 4096},
    )
    return json.loads(resp["message"]["content"])