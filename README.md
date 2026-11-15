# Prompt Engineering and Hallucination Study

A reproducible experiment that measures how five prompting strategies affect the **accuracy** and **hallucination rate** of a Large Language Model (Google Gemini) on a 40-question factual question-answering benchmark.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-gemini--3.5--flash--lite-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## Table of Contents

1. [Overview](#overview)
2. [Prompting Strategies](#prompting-strategies)
3. [Dataset](#dataset)
4. [Hallucination Definition](#hallucination-definition)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Key Findings and Trade-offs](#key-findings-and-trade-offs)
10. [Limitations](#limitations)
11. [Tech Stack](#tech-stack)
12. [Author](#author)

---

## Overview

LLMs can produce fluent but false answers (hallucinations). This project quantifies how much the **way we prompt** a model changes that behaviour. Every question is asked under five different prompting strategies (200 API calls in total), each response is automatically graded as **Correct**, **Hallucination** or **Abstained**, and the results are aggregated into a comparison table and chart.

**Deliverables**

| Item | File |
|---|---|
| Code | `main.py` |
| Question set | `questions.json` |
| Raw and graded results | `results/` |
| Report | `Prompt_Hallucination_Study_Report.pdf` |

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| **Zero-shot** | The question only, with no examples. |
| **Few-shot** | Three solved question-answer examples are placed before the question. |
| **Chain-of-thought (CoT)** | The model is asked to reason step by step before answering. |
| **Retrieval-grounded** | The top-2 relevant passages are retrieved with TF-IDF and the model must answer using only that context (otherwise `UNKNOWN`). |
| **Strict "say unknown"** | The model is told to answer only if confident, and to reply `UNKNOWN` if unsure, if the question has a false premise, or if the answer cannot be known. |

All strategies use the same model (`gemini-3.5-flash-lite`) with `temperature = 0` so that differences come from the prompt, not from sampling.

---

## Dataset

`questions.json` contains **40 questions**:

- **30 answerable** questions: easy facts (e.g. capitals, dates, authors) and some obscure ones (e.g. tungsten's atomic number, the 1982 Literature Nobel laureate). Each has accepted ground-truth answers and a short context passage used by the retrieval strategy.
- **10 unanswerable** questions: fictional entities (e.g. a country called "Zorvania"), future events (e.g. the 2038 World Cup winner), and unknowable facts (e.g. Napoleon's breakfast on a specific day). The correct behaviour here is to abstain.

Each entry has the fields: `id`, `question`, `answers`, `type` and `context`.

---

## Hallucination Definition

A response is graded using its **final answer** (the line starting with `Final Answer:`):

| Label | Rule |
|---|---|
| **Correct** | The final answer matches one of the accepted ground-truth answers. |
| **Hallucination** | The model gives a confident answer that is **wrong**, or gives **any concrete answer to an unanswerable question** (a fabricated fact). |
| **Abstained** | The model says it does not know or that the subject does not exist or has not happened (e.g. `UNKNOWN`, "fictional", "has not occurred yet"). Abstaining is **not** a hallucination. |

Metrics reported per strategy: accuracy, hallucination rate, abstention rate, hallucination rate among attempted answers, accuracy on answerable questions, wrongly abstained answerable questions, hallucination rate on unanswerable questions, and average tokens per question.

---

## Project Structure

```
prompt-hallucination-study/
├── main.py                 # Runs all strategies, grades responses, builds table and chart
├── questions.json          # 40-question benchmark (30 answerable + 10 unanswerable)
├── check_api.py            # Helper: verifies API key and model availability
├── quick_test.py           # Helper: 2-call smoke test before the full run
├── inspect_results.py      # Helper: prints hallucinated / abstained responses for manual review
├── requirements.txt        # Python dependencies
├── .env.example            # Template for the API key file
├── .gitignore              # Excludes .env and venv/
├── README.md
├── Prompt_Hallucination_Study_Report.pdf
└── results/
    ├── raw_responses.csv       # Raw model output for all 200 calls
    ├── graded_responses.csv    # Every response with its final answer and label
    ├── results_table.csv       # Final metrics table
    ├── results_table.md        # Same table in Markdown
    └── results_chart.png       # Accuracy / hallucination / abstention bar chart
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/prompt-hallucination-study.git
cd prompt-hallucination-study
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
google-genai
python-dotenv
pandas
scikit-learn
tqdm
tenacity
matplotlib
```

**4. Add your Gemini API key**

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey), then create a file named `.env` in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

(`.env.example` shows the expected format. Never commit your real `.env` file.)

---

## How to Run

**Step 1: check the API key and model (optional but recommended)**

```bash
python check_api.py
```

If `gemini-3.5-flash-lite` is unavailable for your account, change the `MODEL` variable in `main.py` to another model from the printed list.

**Step 2: run a small smoke test (optional)**

```bash
python quick_test.py
```

**Step 3: run the full experiment**

```bash
python main.py
```

- 200 API calls are made (40 questions x 5 strategies) with a short pause between calls to respect free-tier rate limits. The run takes roughly 25 to 30 minutes.
- Progress is saved after every call to `results/raw_responses.csv`. If the run stops (network, quota), run `python main.py` again and it **resumes** with only the missing calls.
- When all calls are done, responses are graded and the table and chart are written to `results/`.

**Step 4: review flagged responses (optional)**

```bash
python inspect_results.py
```

Grading is keyword-based, so reading the hallucinated and abstained responses is a good way to confirm the labels.

---

## Results

Model: `gemini-3.5-flash-lite`, temperature 0, N = 40 questions per strategy.

| Strategy | Accuracy | Hallucination | Abstention | Halluc. among attempted | Halluc. on unanswerable | Avg tokens / question |
|---|---|---|---|---|---|---|
| Zero-shot | 75.0% | 5.0% | 20.0% | 6.2% | 20% | 78.8 |
| Few-shot | 75.0% | 2.5% | 22.5% | 3.2% | 10% | 97.8 |
| Chain-of-thought | 75.0% | 5.0% | 20.0% | 6.2% | 20% | 192.6 |
| Retrieval-grounded | 75.0% | 0.0% | 25.0% | 0.0% | 0% | 150.5 |
| Strict "say unknown" | 75.0% | 0.0% | 25.0% | 0.0% | 0% | 137.0 |

> **Note on accuracy:** accuracy is computed over all 40 questions and only counts correct answers to the 30 answerable ones. A correct abstention on an unanswerable question is counted under *Abstention*, not *Accuracy*, so the maximum possible accuracy is 75%. All five strategies answered all 30 answerable questions correctly.

The chart is saved at `results/results_chart.png`.

---

## Key Findings and Trade-offs

- **Hallucination only appears on unanswerable questions.** Every strategy answered all 30 answerable questions correctly, so the strategies differ only in how they behave when the right answer is "I don't know".
- **Strict instruction and retrieval grounding gave 0% hallucination** and did not cost any accuracy on this question set, because the model never abstained on an answerable question.
- **Chain-of-thought did not reduce hallucination** compared with zero-shot, and used about 2.4x more tokens (192.6 vs 78.8). Reasoning does not help when the model lacks the fact.
- **Few-shot** slightly reduced hallucination (one fewer error), but with 40 questions this difference is a single response and should not be over-interpreted.
- **Cost vs. reliability:** zero-shot is the cheapest but the least safe. The strict prompt is a cheap way to improve reliability (about 1.7x the tokens of zero-shot). Retrieval is reliable but needs a knowledge base and extra setup.
- **Observed hallucinations** were a made-up parrot name (e.g. "Bubi") and a confident population number of `0` for Faisalabad in 1850.

---

## Limitations

- Small benchmark: with 40 questions, one response equals 2.5 percentage points.
- The answerable questions are easy for a modern LLM, so the usual trade-off (strict prompts causing wrong abstentions on answerable questions) was not visible here. A harder question set would be needed to measure it.
- The retrieval passages come from the same file as the questions, which favours the retrieval strategy.
- Grading is keyword-based. It was refined after manual review of flagged responses, but borderline cases (such as the Faisalabad population question) remain subjective.
- Results come from a single model and a single run at temperature 0.

---

## Tech Stack

Python, Google Gemini API (`google-genai`), pandas, scikit-learn (TF-IDF retrieval), tenacity (retry logic), matplotlib, tqdm, python-dotenv.

---

## Author

**Muhammad Faizan Farooq**, Machine Learning and AI Engineer
GitHub: [faizanjutt473](https://github.com/faizanjutt473)
