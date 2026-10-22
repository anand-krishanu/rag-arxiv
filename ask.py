import sys
from search import Searcher
from generate import answer

searcher = Searcher()
question = " ".join(sys.argv[1:])
chunks = searcher.search([question], method="bm25_rerank", k=5)[0]

result = answer(question, chunks)

if result["found"]:
    print(result["answer"])
    print("cited sources:", result["sources"])
else:
    print("I don't know")

print()
for i, c in enumerate(chunks, 1):
    print(f"[{i}] {c['paper_id']} page {c['page'] + 1}")