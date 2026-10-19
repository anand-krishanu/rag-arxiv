import os
from dotenv import load_dotenv
load_dotenv()
for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
    k = os.environ.get(name)
    print(name, "->", None if k is None else (len(k), k[:4], repr(k[-1])))