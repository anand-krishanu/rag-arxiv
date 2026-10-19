import json, os, random, ollama

def looks_like_references(t):
    return t.count("et al") + t.count("arXiv") + t.count("preprint") > 5

used = {(json.loads(l)["paper_id"], json.loads(l)["page"]) for l in open("draft.jsonl")}

papers = [json.load(open(f"texts/{f}")) for f in os.listdir("texts")]
pairs = []
for p in papers:
    for i, t in enumerate(p["pages"]):
        if len(t) > 800 and not looks_like_references(t) and (p["id"], i) not in used:
            pairs.append((p["id"], i, t))

random.seed(1)
sample = random.sample(pairs, 150)

PROMPT = (
    "Write ONE specific question and a short answer, using only this text. "
    "The question must make sense on its own, for someone who has not seen the text. "
    "Never say 'the text', 'the paper' or 'the passage'. "
    "Do not ask about equations, figures, tables, authors or references. "
    "Reply as JSON with keys question and answer, both plain strings.\n\n"
)

with open("draft2.jsonl", "w") as out:
    for n, (pid, page, text) in enumerate(sample):
        try:
            resp = ollama.chat(
                model="mistral:latest", format="json",
                options={"num_ctx": 4096, "temperature": 0.3},
                messages=[{"role": "user", "content": PROMPT + text}],
            )
            data = json.loads(resp["message"]["content"])
            row = {"paper_id": pid, "page": page,
                   "question": str(data["question"]), "answer": str(data["answer"]),
                   "type": "", "keep": None}
            out.write(json.dumps(row) + "\n")
            print(n, end=" ", flush=True)
        except Exception as e:
            print(f"\nskip {pid} p{page}: {e}")