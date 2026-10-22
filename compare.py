"""One table with every saved run. Usage: python compare.py [--type vague]"""
import argparse, glob, json, os
from metrics import recall_at_k, mrr

ap = argparse.ArgumentParser()
ap.add_argument("--type", help="only this question type: simple, exact-term, vague, multi-hop")
args = ap.parse_args()

print(f"{'run':<22}{'n':>4}{'r@1':>7}{'r@3':>7}{'r@5':>7}{'r@10':>7}{'MRR':>7}{'paper@5':>9}{'ms/query':>10}")
for path in sorted(glob.glob("results/*.json")):
    data = json.load(open(path))
    if isinstance(data, list):  # older files had no latency
        data = {"latency_ms": None, "results": data}
    res = [r for r in data["results"] if not args.type or r["type"] == args.type]
    if not res or "paper_rank" not in res[0]:  # skip empty or very old result files
        continue
    ranks = [r["rank"] for r in res]
    paper = [r["paper_rank"] for r in res]
    ms = "-" if data["latency_ms"] is None else data["latency_ms"]
    name = os.path.basename(path)[:-5]
    print(f"{name:<22}{len(res):>4}{recall_at_k(ranks, 1):>7.2f}{recall_at_k(ranks, 3):>7.2f}"
          f"{recall_at_k(ranks, 5):>7.2f}{recall_at_k(ranks, 10):>7.2f}{mrr(ranks):>7.2f}"
          f"{recall_at_k(paper, 5):>9.2f}{ms:>10}")
