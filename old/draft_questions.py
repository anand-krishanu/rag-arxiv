import json, os, random, ollama

papers = [json.load(open(f"texts/{f}")) for f in os.listdir("texts")]

pairs = []
for p in papers:
    for i, t in enumerate(p["pages"]):
        if len(t) > 800 and "References" not in t[:200]:
            pairs.append((p["id"], i, t))

random.seed(0)
sample = random.sample(pairs, 150)

with open("draft.jsonl", "w") as out:
    for n, (pid, page, text) in enumerate(sample):
        try:
            resp = ollama.chat(
                model="mistral:latest",
                format="json",
                options={"num_ctx": 4096, "temperature": 0.3},
                messages=[{"role": "user", "content":
                    "Write ONE specific question that can be answered only from this text, "
                    "and its short answer. Do not ask general questions. "
                    "Reply as JSON with keys question and answer.\n\n" + text}],
            )
            data = json.loads(resp["message"]["content"])
            row = {"paper_id": pid, "page": page,
                   "question": data["question"], "answer": data["answer"],
                   "type": "", "keep": None}
            out.write(json.dumps(row) + "\n")
            print(n, end=" ", flush=True)
        except Exception as e:
            print(f"\nskip {pid} p{page}: {e}")