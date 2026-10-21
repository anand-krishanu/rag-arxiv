import json, os, random, re, time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def looks_like_references(t):
    return t.count("et al") + t.count("arXiv") + t.count("preprint") > 5

papers = [json.load(open(f"texts/{f}")) for f in os.listdir("texts")]
pairs = []
for p in papers:
    for i, t in enumerate(p["pages"]):
        if len(t) > 800 and not looks_like_references(t):
            pairs.append((p["id"], i, t))
random.seed(3)
sample = random.sample(pairs, 220)

PROMPT = (
    "Write ONE specific question and a short answer, using only this text. "
    "The question must make sense on its own, for someone who has not seen the text. "
    "Never say 'the text', 'the paper' or 'the passage'. Name the method or model in the question. "
    "Do not ask about equations, figures, tables, authors or references. "
    "The answer must be one sentence, a number, or a name. "
    "If the text has no good fact to ask about, reply with question set to an empty string. "
    "Reply as JSON with keys question and answer.\n\n"
)

with open("draf4.jsonl", "w") as out:
    for n, (pid, page, text) in enumerate(sample):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=PROMPT + text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.3),
            )
            data = json.loads(resp.text)
            if not str(data["question"]).strip():
                continue
            row = {"paper_id": pid, "page": page,
                   "question": str(data["question"]), "answer": str(data["answer"]),
                   "type": "", "keep": None}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(n, end=" ", flush=True)
        except Exception as e:
            print(f"\nskip {pid} p{page}: {e}")
            errors += 1
            if errors >= 5:
                raise SystemExit("Too many errors, stopping.")
        time.sleep(4)