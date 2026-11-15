"""
Prompt Engineering and Hallucination Study
------------------------------------------
Compares 5 prompting strategies (zero-shot, few-shot, chain-of-thought,
retrieval-grounded, strict "say unknown") on 40 factual questions using
the Gemini free tier, and measures accuracy / hallucination / abstention.

Run:  python main.py
"""

import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
RESULTS_DIR = BASE_DIR / "results"
RAW_FILE = RESULTS_DIR / "raw_responses.csv"
GRADED_FILE = RESULTS_DIR / "graded_responses.csv"
TABLE_CSV = RESULTS_DIR / "results_table.csv"
TABLE_MD = RESULTS_DIR / "results_table.md"
CHART_FILE = RESULTS_DIR / "results_chart.png"

MODEL = "gemini-3.5-flash-lite"
SLEEP_BETWEEN_CALLS = 6.5       # free tier rate limit (~10 requests/min) ke liye
TOP_K = 2                       # retrieval: kitne passages dene hain

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought", "retrieval_grounded", "strict_unknown"]

FORMAT_INSTRUCTION = (
    "Keep your answer short. End your reply with a final line in exactly this format:\n"
    "Final Answer: <your answer>"
)

# ----------------------------------------------------------------------
# Prompt builders (one per strategy)
# ----------------------------------------------------------------------
FEW_SHOT_EXAMPLES = (
    "Question: What is the capital city of France?\nFinal Answer: Paris\n\n"
    "Question: Who wrote the play 'Romeo and Juliet'?\nFinal Answer: William Shakespeare\n\n"
    "Question: What is the chemical symbol for sodium?\nFinal Answer: Na\n\n"
)


def prompt_zero_shot(question, context=None):
    return f"Answer the following factual question.\n\nQuestion: {question}\n\n{FORMAT_INSTRUCTION}"


def prompt_few_shot(question, context=None):
    return (
        "Answer factual questions. Here are some solved examples:\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        f"Now answer this question in the same style.\n\nQuestion: {question}\nFinal Answer:"
    )


def prompt_cot(question, context=None):
    return (
        "Answer the following factual question. Think step by step and reason carefully "
        "before giving your answer.\n\n"
        f"Question: {question}\n\n"
        "After your reasoning, end with a final line in exactly this format:\n"
        "Final Answer: <your answer>"
    )


def prompt_retrieval(question, context=None):
    return (
        "Answer the question using ONLY the context passages below. "
        "If the context does not contain the answer, the final answer must be UNKNOWN.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n{FORMAT_INSTRUCTION}"
    )


def prompt_strict(question, context=None):
    return (
        "Answer the following factual question. Only answer if you are confident that your "
        "answer is correct. If you are not sure, if the question has a false premise, or if "
        "the answer cannot be known, the final answer must be exactly UNKNOWN. "
        "Do not guess.\n\n"
        f"Question: {question}\n\n"
        "End your reply with a final line in exactly this format:\n"
        "Final Answer: <your answer or UNKNOWN>"
    )


PROMPT_FUNCS = {
    "zero_shot": prompt_zero_shot,
    "few_shot": prompt_few_shot,
    "chain_of_thought": prompt_cot,
    "retrieval_grounded": prompt_retrieval,
    "strict_unknown": prompt_strict,
}

# ----------------------------------------------------------------------
# Retrieval (TF-IDF over the passages in questions.json)
# ----------------------------------------------------------------------
class Retriever:
    def __init__(self, questions):
        self.docs = [q["context"] for q in questions if q["context"]]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def get_context(self, question, k=TOP_K):
        q_vec = self.vectorizer.transform([question])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        top_idx = scores.argsort()[::-1][:k]
        return "\n".join(f"[{i + 1}] {self.docs[j]}" for i, j in enumerate(top_idx))


# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------
def _is_transient(e):
    # Sirf server busy / rate limit errors par retry karo, 400 par nahi
    return getattr(e, "code", None) in (429, 500, 503, 504)


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    reraise=True,
)
def call_llm(client, prompt):
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096),
    )
    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    p_tok = getattr(usage, "prompt_token_count", 0) or 0
    o_tok = getattr(usage, "candidates_token_count", 0) or 0
    return text.strip(), p_tok, o_tok


# ----------------------------------------------------------------------
# Running all questions x strategies (with resume support)
# ----------------------------------------------------------------------
def load_cache():
    if RAW_FILE.exists():
        df = pd.read_csv(RAW_FILE, keep_default_na=False)
        return {(int(r["id"]), r["strategy"]) for _, r in df.iterrows()}
    return set()


def run_experiment(client, questions):
    RESULTS_DIR.mkdir(exist_ok=True)
    retriever = Retriever(questions)
    done = load_cache()

    new_file = not RAW_FILE.exists()
    f = open(RAW_FILE, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["id", "strategy", "prompt_tokens", "output_tokens", "response"])
        f.flush()

    jobs = [(q, s) for q in questions for s in STRATEGIES if (q["id"], s) not in done]
    print(f"Total jobs baqi: {len(jobs)} (pehle se complete: {len(done)})")

    for q, strategy in tqdm(jobs, desc="Calling Gemini"):
        context = retriever.get_context(q["question"]) if strategy == "retrieval_grounded" else None
        prompt = PROMPT_FUNCS[strategy](q["question"], context)
        try:
            text, p_tok, o_tok = call_llm(client, prompt)
        except Exception as e:
            print(f"\nError (id={q['id']}, {strategy}): {e}\n"
                  f"Ye skip hua, script dobara chalao to sirf baqi calls hongi.")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue
        writer.writerow([q["id"], strategy, p_tok, o_tok, text])
        f.flush()
        time.sleep(SLEEP_BETWEEN_CALLS)
    f.close()


# ----------------------------------------------------------------------
# Grading: Correct / Hallucination / Abstained
# ----------------------------------------------------------------------
def normalize(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


ABSTAIN_PHRASES = [
    "unknown", "i don't know", "i do not know", "not sure", "no one", "nobody", "none",
    "not aware", "no record", "no historical record", "no public record", "no evidence",
    "no reliable", "no known", "no documented", "not documented", "no surviving",
    "does not exist", "doesn't exist", "fictional", "fictitious", "nonexistent",
    "cannot be determined", "cannot determine", "can't determine", "not possible to",
    "no information", "not yet", "unable to", "no such", "cannot answer", "can't answer",
    "not provided", "not mentioned", "not available", "no human", "no person",
    "has not happened", "has not occurred", "has not been", "have not been",
    "hasn't happened", "hasn't occurred", "never happened", "not happened", "not occurred",
]
ABSTAIN_NORM = [normalize(p) for p in ABSTAIN_PHRASES]


def extract_final_answer(response):
    matches = re.findall(r"final answer\s*:\s*(.*)", response, flags=re.IGNORECASE)
    if matches:
        ans = matches[-1].strip().strip("*").strip()
        if ans:
            return ans
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def is_abstain(final_answer):
    if final_answer.strip() == "":
        return True
    n = normalize(final_answer)
    return any(p in n for p in ABSTAIN_NORM)


def matches_truth(final_answer, accepted):
    n = normalize(final_answer)
    return any(normalize(a) in n for a in accepted)


def grade(question, response):
    final = extract_final_answer(response)
    if is_abstain(final):
        return final, "abstained"
    if question["type"] == "unanswerable":
        return final, "hallucination"      # model ne confident answer diya jo ho hi nahi sakta
    if matches_truth(final, question["answers"]):
        return final, "correct"
    return final, "hallucination"          # confident but ghalat answer


def grade_all(questions):
    raw = pd.read_csv(RAW_FILE, keep_default_na=False)
    qmap = {q["id"]: q for q in questions}
    rows = []
    for _, r in raw.iterrows():
        q = qmap[int(r["id"])]
        final, label = grade(q, str(r["response"]))
        rows.append({
            "id": int(r["id"]),
            "type": q["type"],
            "question": q["question"],
            "ground_truth": " | ".join(q["answers"]),
            "strategy": r["strategy"],
            "final_answer": final,
            "label": label,
            "prompt_tokens": int(r["prompt_tokens"]),
            "output_tokens": int(r["output_tokens"]),
            "full_response": r["response"],
        })
    graded = pd.DataFrame(rows).sort_values(["strategy", "id"])
    graded.to_csv(GRADED_FILE, index=False, encoding="utf-8")
    return graded


# ----------------------------------------------------------------------
# Metrics + table + chart
# ----------------------------------------------------------------------
def pct(x):
    return round(100.0 * x, 1)


def compute_table(graded):
    rows = []
    for s in STRATEGIES:
        g = graded[graded["strategy"] == s]
        if g.empty:
            continue
        n = len(g)
        ans = g[g["type"] == "answerable"]
        unans = g[g["type"] == "unanswerable"]
        attempted = g[g["label"] != "abstained"]
        rows.append({
            "Strategy": s,
            "N": n,
            "Accuracy %": pct((g["label"] == "correct").mean()),
            "Hallucination %": pct((g["label"] == "hallucination").mean()),
            "Abstention %": pct((g["label"] == "abstained").mean()),
            "Halluc. among attempted %": pct((attempted["label"] == "hallucination").mean()) if len(attempted) else 0.0,
            "Answerable acc %": pct((ans["label"] == "correct").mean()) if len(ans) else 0.0,
            "Wrongly abstained (answerable) %": pct((ans["label"] == "abstained").mean()) if len(ans) else 0.0,
            "Unanswerable halluc %": pct((unans["label"] == "hallucination").mean()) if len(unans) else 0.0,
            "Avg tokens/question": round((g["prompt_tokens"] + g["output_tokens"]).mean(), 1),
        })
    return pd.DataFrame(rows)


def save_markdown(table):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")


def save_chart(table):
    x = range(len(table))
    w = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar([i - w for i in x], table["Accuracy %"], w, label="Accuracy")
    plt.bar(list(x), table["Hallucination %"], w, label="Hallucination")
    plt.bar([i + w for i in x], table["Abstention %"], w, label="Abstention")
    plt.xticks(list(x), table["Strategy"], rotation=15)
    plt.ylabel("Percentage (%)")
    plt.title("Prompting Strategy vs Accuracy / Hallucination / Abstention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY nahi mili. .env file check karo.")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    assert len(questions) >= 40, "Kam az kam 40 questions chahiye"

    client = genai.Client(api_key=api_key)

    run_experiment(client, questions)

    if not RAW_FILE.exists():
        raise SystemExit("Koi response save nahi hua.")

    graded = grade_all(questions)
    table = compute_table(graded)
    table.to_csv(TABLE_CSV, index=False)
    save_markdown(table)
    save_chart(table)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n===== RESULTS =====")
    print(table.to_string(index=False))
    print(f"\nFiles save hui hain: {RESULTS_DIR}")

    if len(graded) < len(questions) * len(STRATEGIES):
        print("\nWarning: kuch calls fail hui thi. main.py dobara chalao, baqi calls complete ho jayengi.")


if __name__ == "__main__":
    main()