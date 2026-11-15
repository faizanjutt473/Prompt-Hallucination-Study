import pandas as pd

df = pd.read_csv("results/graded_responses.csv", keep_default_na=False)

print("=== HALLUCINATION wale responses ===")
h = df[df["label"] == "hallucination"]
for _, r in h.iterrows():
    print(f"[{r['strategy']}] Q{r['id']}: {r['question']}")
    print(f"    Final answer: {r['final_answer']}\n")

print("=== Unanswerable par ABSTAINED responses (sirf zero_shot, few_shot, chain_of_thought) ===")
a = df[(df["label"] == "abstained") & (df["type"] == "unanswerable")
       & (df["strategy"].isin(["zero_shot", "few_shot", "chain_of_thought"]))]
for _, r in a.iterrows():
    print(f"[{r['strategy']}] Q{r['id']}: {r['final_answer']}")