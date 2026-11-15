import os
from dotenv import load_dotenv
from google import genai
import main

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

tests = [
    ("zero_shot", "What is the capital city of Australia?"),
    ("strict_unknown", "What is the capital city of the country of Zorvania?"),
]

for strategy, question in tests:
    prompt = main.PROMPT_FUNCS[strategy](question, None)
    text, p, o = main.call_llm(client, prompt)
    print(f"--- {strategy} ---")
    print(text)
    print("Final:", main.extract_final_answer(text))
    print("Tokens:", p, o, "\n")