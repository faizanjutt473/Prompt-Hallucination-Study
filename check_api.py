import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("GEMINI_API_KEY nahi mili. .env file check karo.")

print("API key mili, length:", len(api_key))

client = genai.Client(api_key=api_key)

# 1) Available models ki list
print("\nAvailable models (generateContent support wale):")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(" -", m.name)

# 2) Model ka test call
MODEL = "gemini-3.5-flash-lite"
try:
    resp = client.models.generate_content(
        model=MODEL,
        contents="Reply with one word: Hello",
    )
    print(f"\n{MODEL} kaam kar raha hai. Response:", resp.text)
except Exception as e:
    print(f"\n{MODEL} me error aaya:", e)