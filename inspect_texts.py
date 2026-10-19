import json, os
sizes = []
for f in os.listdir("texts"):
    d = json.load(open(f"texts/{f}"))
    sizes.append((f, len(d["pages"]), sum(len(p) for p in d["pages"])))
print("papers:", len(sizes))
print("total pages:", sum(s[1] for s in sizes))
print("papers under 5000 chars:", [s[0] for s in sizes if s[2] < 5000])