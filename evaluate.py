import argparse, json, os, time
from collections import defaultdict
from metrics import first_hit_rank, recall_at_k, mrr, all_gold_in_top_k
from search import Searcher

KS = [1, 3, 5, 10]

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="dense",
                choices=["dense", "bm25", "hybrid", "bm25_rerank", "hybrid_rerank", "dense_rerank"])
ap.add_argument("--rerank-pool", type=int, default=30, help="how many first-stage results the reranker re-sorts")
ap.add_argument("--words", type=int, default=200)
ap.add_argument("--overlap", type=int, default=50)
ap.add_argument("--testset", default="testset.jsonl")
ap.add_argument("--final", action="store_true", help="score the final test set (use ONCE at the end)")
ap.add_argument("--show", type=int, default=10, help="how many failed questions to print")
args = ap.parse_args()

# --- load questions
questions = []
for line in open(args.testset, encoding="utf-8"):
    r = json.loads(line)
    if r.get("type") == "no-answer" or bool(r.get("final_test")) != args.final:
        continue
    questions.append(r)
print(f"{len(questions)} questions ({'FINAL' if args.final else 'dev'} set), method={args.method}, "
      f"chunks={args.words}/{args.overlap}")

# --- search
searcher = Searcher(args.words, args.overlap)
searcher.search(["warm up"], method=args.method, k=max(KS), rerank_pool=args.rerank_pool)  # load models first
t0 = time.time()
retrieved = searcher.search([q["question"] for q in questions], method=args.method, k=max(KS),
                            rerank_pool=args.rerank_pool)
latency_ms = (time.time() - t0) / len(questions) * 1000
print(f"average search time: {latency_ms:.0f} ms per query")

# --- score (page level: right paper + right page +-1; paper level: right paper only)
results = []
for q, chunks in zip(questions, retrieved):
    results.append({"q": q, "retrieved": chunks,
                    "rank": first_hit_rank(chunks, q["gold"]),
                    "paper_rank": first_hit_rank(chunks, q["gold"], page_slack=10**6),
                    "all_gold": all_gold_in_top_k(chunks, q["gold"], max(KS))})


def line(name, rs, key="rank"):
    ranks = [r[key] for r in rs]
    rec = "  ".join(f"recall@{k}={recall_at_k(ranks, k):.2f}" for k in KS)
    return f"{name:<12} n={len(rs):<3} {rec}  MRR={mrr(ranks):.2f}"


print("\n== PAGE level (right paper AND right page)")
print(line("ALL", results))
by_type = defaultdict(list)
for r in results:
    by_type[r["q"]["type"]].append(r)
for t, rs in sorted(by_type.items()):
    print(line(t, rs))

print("\n== PAPER level (right paper, any page)")
print(line("ALL", results, "paper_rank"))
for t, rs in sorted(by_type.items()):
    print(line(t, rs, "paper_rank"))

multi = [r for r in results if len(r["q"]["gold"]) > 1]
if multi:
    print(f"\nmulti-hop: all gold places in top {max(KS)}: {sum(r['all_gold'] for r in multi)}/{len(multi)}")

# --- failures: split into 'wrong paper' and 'right paper, wrong page'
wrong_paper = [r for r in results if r["paper_rank"] is None]
wrong_page = [r for r in results if r["paper_rank"] is not None and r["rank"] is None]
print(f"\nno hit in top {max(KS)}: {len(wrong_paper)} wrong paper, {len(wrong_page)} right paper but wrong page")
for r in wrong_paper[: args.show]:
    g, top = r["q"]["gold"][0], r["retrieved"][0]
    print(f"- [{r['q']['type']}] {r['q']['question']}")
    print(f"    gold: {g['paper_id']} p{g['page'] + 1}   top result: {top['paper_id']} p{top['page'] + 1}")

os.makedirs("results", exist_ok=True)
out = {"method": args.method, "latency_ms": round(latency_ms),
       "results": [{"id": r["q"]["id"], "type": r["q"]["type"], "rank": r["rank"], "paper_rank": r["paper_rank"]}
                   for r in results]}
suffix = f"_pool{args.rerank_pool}" if args.method.endswith("_rerank") else ""
json.dump(out, open(f"results/{args.method}{suffix}_{args.words}_{args.overlap}.json", "w"), indent=1)
