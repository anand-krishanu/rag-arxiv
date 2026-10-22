import json
from search import Searcher
from generate import answer

searcher = Searcher()
rows = [json.loads(line) for line in open("testset.jsonl", encoding="utf-8")]
rows = [r for r in rows if not r["final_test"]]      # dev questions only, never the hidden ones

all_chunks = searcher.search([r["question"] for r in rows], method="bm25_rerank", k=5)

out = []
for r, chunks in zip(rows, all_chunks):
    res = answer(r["question"], chunks)
    out.append({
        "id": r["id"], "type": r["type"], "question": r["question"],
        "gold_answer": r["answer"],
        "found": res["found"], "answer": res["answer"], "cited": res["sources"],
        "chunks": [{"paper_id": c["paper_id"], "page": c["page"], "text": c["text"]} for c in chunks],
    })
    print(len(out), end=" ", flush=True)

with open("results/answers.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")