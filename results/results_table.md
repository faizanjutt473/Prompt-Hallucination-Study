| Strategy | N | Accuracy % | Hallucination % | Abstention % | Halluc. among attempted % | Answerable acc % | Wrongly abstained (answerable) % | Unanswerable halluc % | Avg tokens/question |
|---|---|---|---|---|---|---|---|---|---|
| zero_shot | 40 | 75.0 | 5.0 | 20.0 | 6.2 | 100.0 | 0.0 | 20.0 | 78.8 |
| few_shot | 40 | 75.0 | 2.5 | 22.5 | 3.2 | 100.0 | 0.0 | 10.0 | 97.8 |
| chain_of_thought | 40 | 75.0 | 5.0 | 20.0 | 6.2 | 100.0 | 0.0 | 20.0 | 192.6 |
| retrieval_grounded | 40 | 75.0 | 0.0 | 25.0 | 0.0 | 100.0 | 0.0 | 0.0 | 150.5 |
| strict_unknown | 40 | 75.0 | 0.0 | 25.0 | 0.0 | 100.0 | 0.0 | 0.0 | 137.0 |