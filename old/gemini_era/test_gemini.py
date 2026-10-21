from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
resp = client.models.generate_content(model="gemini-2.5-flash", contents="Say hi in 3 words")
print(resp.text)