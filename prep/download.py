import arxiv, os, requests, time

os.makedirs("pdfs", exist_ok=True)
client = arxiv.Client(page_size=100, delay_seconds=3)

queries = [
    'all:"KV cache" AND cat:cs.LG',
    'all:"LLM inference" AND cat:cs.CL',
    'all:quantization AND all:"language model"',
    'all:"speculative decoding"',
    'all:"attention" AND all:"long context" AND all:efficient',
]

seen = set()
for q in queries:
    search = arxiv.Search(query=q, max_results=80, sort_by=arxiv.SortCriterion.Relevance)

    for r in client.results(search):
        pid = r.get_short_id()
        if pid in seen:
            continue
        if len(seen) >= 242:
            break

        path = f"pdfs/{pid}.pdf"
        if not os.path.exists(path):
            try:
                resp = requests.get(r.pdf_url, timeout=60)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                time.sleep(3)
            except Exception as e:
                print("fail", pid, e)
                continue
        seen.add(pid)

print(len(seen), "papers")