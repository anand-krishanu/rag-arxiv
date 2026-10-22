import argparse, json, os, pickle
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

ap = argparse.ArgumentParser()
ap.add_argument("--words", type=int, default=200, help="words per chunk")
ap.add_argument("--overlap", type=int, default=50, help="words shared between neighbour chunks")
args = ap.parse_args()

# 1. Cut every page into overlapping chunks. Each chunk remembers its paper and page.
chunks = []
step = args.words - args.overlap
for f in sorted(os.listdir("texts")):
    d = json.load(open(f"texts/{f}", encoding="utf-8"))
    for page_no, text in enumerate(d["pages"]):
        words = text.split()
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + args.words])
            if len(piece) > 100:
                chunks.append({"paper_id": d["id"], "page": page_no, "text": piece})
print(len(chunks), "chunks")

# 2. Turn every chunk into a vector on the GPU.
model = SentenceTransformer(MODEL_NAME, device="cuda")
emb = model.encode([c["text"] for c in chunks], batch_size=64,
                   normalize_embeddings=True, show_progress_bar=True)

# 3. Store the vectors in FAISS (inner product on unit vectors = cosine similarity).
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

os.makedirs("indexes", exist_ok=True)
tag = f"{args.words}_{args.overlap}"
faiss.write_index(index, f"indexes/{tag}.faiss")
pickle.dump(chunks, open(f"indexes/{tag}.pkl", "wb"))
print("saved indexes/" + tag + ".faiss and .pkl")
