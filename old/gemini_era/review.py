import json, os, re

rows = []
for f in ["draft3.jsonl"]:
    rows += [json.loads(l) for l in open(f)]

done = set()
if os.path.exists("reviewed.jsonl"):
    done = {(r["paper_id"], r["page"]) for r in map(json.loads, open("reviewed.jsonl"))}

BAD = re.compile(r"\b(the text|the paper|equation|figures?|tables?|authors?|et al)\b")

def get_page(r):
    return json.load(open(f"texts/{r['paper_id']}.json"))["pages"][r["page"]]

def obviously_bad(r):
    q = str(r["question"]).lower()
    if BAD.search(q):
        return True
    page = get_page(r).lower()
    words = [w for w in str(r["answer"]).lower().split() if len(w) > 4]
    if not words:
        return True
    found = sum(w in page for w in words)
    return found / len(words) < 0.5

rows = [r for r in rows if not obviously_bad(r)]
print(len(rows), "questions left to review")

TYPES = {"s": "simple", "e": "exact-term", "v": "vague"}

with open("reviewed.jsonl", "a") as out:
    for r in rows:
        if (r["paper_id"], r["page"]) in done:
            continue
        print("\n" + "=" * 80)
        print(get_page(r)[:4000])
        print("-" * 80)
        print(f"Paper: {r['paper_id']}   Page: {r['page'] + 1}")
        print("Q:", r["question"])
        print("A:", r["answer"])
        choice = input("[s]imple [e]xact-term [v]ague [d]elete [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice in TYPES:
            r["type"] = TYPES[choice]
            r["keep"] = True
        else:
            r["keep"] = False
        out.write(json.dumps(r) + "\n")