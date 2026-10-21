import pymupdf, os, json

os.makedirs("texts", exist_ok=True)
for f in os.listdir("pdfs"):
    try:
        doc = pymupdf.open(f"pdfs/{f}")
        pages = [p.get_text() for p in doc]
    except Exception as e:
        print("fail", f, e)
        continue
    json.dump({"id": f[:-4], "pages": pages}, open(f"texts/{f[:-4]}.json", "w"))