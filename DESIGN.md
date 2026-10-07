# DistillRouter: Research Design

*Final draft, 2026-10-07. Binary routing: small or large.*

---

## 0. Summary

A **router** sends each question to a cheap **small** model or an expensive **large** model.
Good routers are often large LLMs that read the question, reason about it and decide. They are
accurate but **slow**: every question waits for the router's generation.

We **distil** a large LLM router (teacher) into a small model (student) that reads the question
once and makes the same decision in milliseconds. This is SFT-style distillation: the teacher
labels questions, the student learns those labels with cross-entropy.

| Part | Choice |
|---|---|
| Candidates | Small Qwen2.5-1.5B-Instruct, large Qwen2.5-14B-Instruct |
| Teacher | Qwen2.5-14B-Instruct: short explanation, then small/large; 5 votes → soft label |
| Student | Gemma-3 270M / 1B: route-only, and reasoning-plus-route (also learns the explanation) |
| Oracle | Both candidates answer 5×; answers checked against references; grading only |
| Data | GSM8K, MATH (2,000 / 300 / 500 each) + GSM-Symbolic test (500) |
| Main result | Latency-quality curve: teacher vs students |

---

## 1. Problem

> **Can the routing policy of a larger teacher model be distilled into a lightweight model?**
>
> *Given a teacher routing policy π_T, can we learn a lightweight routing policy π_θ that
> preserves routing accuracy while significantly reducing routing latency and computational overhead?*

*Update `research.tex` to this wording ("an expensive model" → "a larger teacher model").*

---

## 2. Research questions

| RQ | Question (`research.tex`) | In plain words |
|---|---|---|
| RQ1 | How effectively can a routing policy be transferred from an accurate but expensive teacher router to a lightweight distilled router? | How well and how fast does the student copy the teacher? |
| RQ2 | Does augmenting routing supervision with teacher reasoning improve policy transfer relative to route-only distillation? | Does also learning the teacher's explanation help? |
| RQ3 | Do improvements in policy transfer translate into improvements in routing accuracy under independent oracle labels? | Does better copying give better routing? |

Copying (fidelity) and correctness are reported separately; they need not move together
(Stanton et al., 2021).

---

## 3. Design principles

| # | Principle | Source | Where used |
|---|---|---|---|
| P1 | **Copying ≠ correct.** Agreement with the teacher measures imitation; correctness needs an independent answer key | Stanton et al. 2021; Wu et al. 2026 | Oracle grading, RQ3 (§4.4) |
| P2 | **Label quality decides student quality.** Clean labels beat more labels | Course notes; Wu et al. | Teacher check (§10); E7 |
| P3 | **Soft labels must be a full distribution** | Hinton et al. 2015; Wu et al. | 5-vote share over both labels (§4.2) |
| P4 | **Combine hard and soft supervision** | Wu et al. | Loss options (§4.3, E6) |
| P5 | **Train on what you decide.** Training target and deployed decision must match | *When Routing Collapses* (2026) | Student threshold 0.5 = teacher's majority rule |
| P6 | **Check other data.** Routers are sensitive to distribution shift | Wu et al.; zero-shot confidence study (2026) | GSM-Symbolic (E8) |
| P7 | **Honest latency.** p50/p95/p99, same serving stack for all routers | Wu et al.; MLPerf; vLLM metrics | §9 |
| P8 | **Test once, fixed seeds, error bars over questions** | Course notes; reviewers | §12 |
| P9 | **Diagnose before fixing.** One score hides different failures; fix only what is broken | Course notes (error analysis) | §11 |

---

## 4. Approach

### 4.1 Candidates

| Role | Model |
|---|---|
| Small | Qwen2.5-1.5B-Instruct |
| Large | Qwen2.5-14B-Instruct (backup: Qwen2.5-7B-Instruct) |

Clear quality gap (GSM8K 73.2 vs 94.8; MATH 55.2 vs 80.0, official Qwen2.5 scores), clear cost
gap (~9× parameters), and a small model that is enough for many questions.
Prompt: *"Please reason step by step, and put your final answer within \boxed{}."*
(Qwen2.5-Math model card). 5 samples, temperature 0.7, top-p 0.95, up to 1,024 tokens, vLLM.

### 4.2 Teacher: a large LLM router

Qwen2.5-14B-Instruct, two steps (`teacher/qwen_teacher.py`):

1. **Explain:** one sentence on what kind of reasoning the question needs (zero-shot).
2. **Decide:** small (a ~1-2B model is enough) or large (a ~14B model is needed), given the
   question, the explanation and a few examples with known outcomes (calibration split).

- **5 votes** (temperature 0.7) → **soft label** = share of votes for "large"; hard label =
  majority. The kept explanation comes from the majority side.
- **Deployable teacher** (baseline and latency): one answer at temperature 0.
- Unreadable answer → ask again twice → "large".

The teacher sees only the question, never the candidates' answers or the answer key, so it works
on real traffic.

### 4.3 Student (SFT-style distillation)

Gemma-3-270M-it and Gemma-3-1B-it read the question and output P(large) in one forward pass.
Full fine-tuning (the students are small; LoRA is not needed).

- **Route-only:** cross-entropy against the teacher's soft label (P3).
- **Reasoning-plus-route:** + next-token loss on the teacher's explanation, question tokens
  masked (training only; same deployment cost).
- **Decision:** large if P(large) ≥ 0.5 (P5).
- **Loss options (E6, P4):** soft only (default) · hard only · hard + soft
  (CE on the majority label + KL on the vote share).
- **Settings:** AdamW, 3 epochs, batch 8, lr 2e-5, linear schedule; best checkpoint by
  validation macro-F1 against the teacher. Training and validation loss saved for every run.
- **One-seed check first** (validation only): 10 epochs; batch 16 with lr 4e-5;
  10% warm-up with weight decay 0.01. Change a setting only if validation clearly improves.

### 4.4 Oracle (grading only, never training)

p̂ = share of a model's 5 answers that are correct (math-verify). A model can solve a question
if p̂ ≥ 0.6.

| Small can solve | Large can solve | Oracle label |
|---|---|---|
| Yes | – | small |
| No | Yes | large |
| No | No | none (unsolvable): left out of routing accuracy |

---

## 5. Data (done)

| Dataset | Train | Validation | Test | Source |
|---|---|---|---|---|
| GSM8K | 2,000 | 300 | 500 | `openai/gsm8k` |
| MATH | 2,000 | 300 | 500 | `EleutherAI/hendrycks_math`; test = `HuggingFaceH4/MATH-500` |
| GSM-Symbolic | – | – | 500 | `apple/GSM-Symbolic` |

Splits keep the source's difficulty and subject mix, do not overlap, and are sealed by
`split_manifest.json`. Calibration (20 per dataset) holds the teacher's examples. Exact
duplicates are removed; a near-duplicate check is added in §10.

---

## 6. Flow

### 6.1 Six steps

```
1. PREPARE    questions              → data/processed/   (done)
2. GENERATE   candidates answer      → data/answers/     (GPU)
3. TEACH      teacher labels         → data/teacher/     (GPU)
4. LABEL      oracle + teacher join  → data/labels/      (no GPU)
5. TRAIN      labels → students      → runs/             (small GPU)
6. EVALUATE   all routers → results  → results/          (test opened only here)
```

| Step | Command |
|---|---|
| 1 | `run.py prepare-data --all` |
| Pilot | `run.py pilot` (50 validation questions: speed, truncation, vote disagreement, 100 answers to hand-check) |
| 2 | `run.py generate --model small --split calibration validation test`; same for `--model large` |
| 3 | `run.py teach --split validation test train` |
| 4 | `run.py label` |
| Check | `run.py check` (stops the flow if §10 fails) |
| 5 | `run.py train --student 270m --variant route_only --seed 0` |
| 6 | `run.py evaluate` (baselines, latency, error analysis, tables, figures) |

### 6.2 Records

```
Question    {id, dataset, split, query, reference_answer, difficulty, metadata}
Answer      {id, model, sample, text, final_answer, correct, in_tokens, out_tokens, seconds, truncated}
Teacher     {id, votes, soft_large, label, greedy_label, explanation, seconds}
Label       {id, dataset, split, p_small, p_large, oracle, teacher, soft_large, explanation}
Prediction  {id, router, p_large, route, seconds}
```

- Students read only `teacher`, `soft_large`, `explanation`; the loader rejects oracle fields.
- Every router writes Predictions, so one evaluation scores them all.

### 6.3 Code

```
common/      config  schema  scoring  generation (vLLM)
dataset/     base  gsm8k  math  gsm_symbolic          (done)
teacher/     qwen_teacher  fewshot  labeler            (add 5 votes)
pipeline/    generate  label  check
student/     data  route_head  train_classifier  train_dual   (add soft-label loss)
evaluation/  predict  baselines  latency  metrics  errors  report
run.py       one command per step
```

### 6.4 Rules

Resumable · versioned (new folder per prompt or setting) · no leaks (students never see the
oracle) · seeded · every paper number produced by step 6 · run `source .venv/bin/activate`
first (vLLM needs `ninja`).

### 6.5 Workload

| Model | Questions | Generations |
|---|---|---|
| Small and large candidates | calibration, validation, test, GSM-Symbolic: 2,140 | 2,140 × 2 × 5 = 21,400 |
| Teacher (14B, two steps) | validation, test, GSM-Symbolic, train: 6,100 | 6,100 × 6 = 36,600 |
| Large on train *(optional, D10)* | 4,000 (or 1,000 subset) | 20,000 (or 5,000) |

---

## 7. Experiments

**Main grid:** students 270M, 1B × route-only, reasoning-plus-route × seeds (D9).

| ID | Experiment | Answers |
|---|---|---|
| E1 | Fidelity and latency of every student | RQ1 |
| E2 | Route-only vs reasoning-plus-route | RQ2 |
| E3 | Routing accuracy, end-to-end accuracy, cost-quality vs teacher and baselines (reported first) | RQ3 |
| E4 | Noisy teacher: 10/20/30% random flips; systematic bias (MATH → large, long → large) | Does the student copy teacher errors? (R1) |
| E5 | Size ladder: TF-IDF, ModernBERT-base, DeBERTa-v3-small, Gemma-270M, Gemma-1B; GPU and CPU latency | How small can the student be? (R2) |

Optional, if time:

| ID | Experiment | Principle |
|---|---|---|
| E6 | Loss: soft only vs hard only vs hard + soft | P3, P4 |
| E7 | Train only on confident teacher labels (≥ 4 of 5 votes) vs all | P2 |
| E8 | GSM-Symbolic test (changed names and numbers) | P6 |
| E9 | Shuffled-explanation control for RQ2 | RQ2 validity |
| E10 | Compiled students (`torch.compile`); student served with vLLM | P7 |

---

## 8. Baselines

| Baseline | Question |
|---|---|
| Always small / always large; random mix at equal cost | Is routing worth it? |
| Teacher, 1 vote (deployable) and 5 votes | Quality kept, latency removed |
| TF-IDF + logistic regression on teacher labels (E5) | Is a neural student needed? |
| Student trained on oracle labels *(optional; D10)* | Gap to having the answer key (R2) |

Same questions, validation-only tuning and latency protocol for all.

---

## 9. Measures

**Latency (primary):** router p50 / p95 / p99; router overhead (% of response time); speedup over
the teacher; setup cost and pay-off. Same GPU, nothing else running, tokenisation included,
20 warm-up questions, then each test question one at a time; teacher on vLLM; students also on CPU.

**Latency-quality curve (headline):** one point per router (teacher, Gemma-1B, Gemma-270M,
DeBERTa, TF-IDF): latency against fidelity and routing accuracy.

**Quality:** fidelity (accuracy, macro-F1 vs teacher); routing accuracy on solvable questions;
end-to-end accuracy (p̂ of the chosen model); cost (2 × parameters × tokens); cost-quality
curve, APGR, CPT(50%), CPT(80%); calibration (ECE) of the student's P(large).

---

## 10. Checks (on validation, before training)

| Check | Pass if | Else |
|---|---|---|
| Speed | All generation projected within 3 days | Backup 7B; skip optional train runs |
| Answer checker | ≥ 95 of 100 agree with a human | Fix extraction |
| Truncation | < 5% hit 1,024 tokens | Raise the limit |
| Solvable | < 20% unsolvable | Change models |
| Enough "large" | ≥ 15% need the large model | Larger large model |
| Vote disagreement | ≥ 10% of questions have split teacher votes | Soft = hard label: drop E6, or raise teacher temperature (validation only) |
| Near-duplicates | No train question has high word overlap with a validation or test question | Remove the train question |
| **Teacher** | Teacher beats random mixing at equal cost | Improve the prompt (≤ 2 revisions, validation only); then report honestly |

---

## 11. Error analysis (validation; test reported once)

A router fails in two ways:

| Error | Meaning | Cost |
|---|---|---|
| Under-routing | Sent to small; small fails, large solves | Quality lost |
| Over-routing | Sent to large; small could solve | Time and money wasted |

1. Count both for the teacher and every student.
2. Break down by dataset, and for MATH by level and subject.
3. Separate the cause: **teacher wrong** (student copied a bad label) vs **student failed to copy**.
4. Read ~50 errors of each kind by hand; group them into a few categories.
5. Act only on a large category, only through the teacher prompt (≤ 2 revisions, validation only).
6. On test: the same breakdown, once, with confidence intervals; small cells are not over-read.

---

## 12. Fairness rules

Test used once · this document committed before test results · error bars over questions and
seeds · Holm correction · "no difference found" when an interval includes zero · results per
dataset and pooled · loss curves for every run in the appendix.

---

## 13. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Binary routing | Final |
| D2 | Candidates Qwen2.5-1.5B / 14B (backup 7B) | Final |
| D3 | Teacher Qwen2.5-14B, two steps, 5 votes, soft labels | Final; checked in §10 |
| D4 | Candidate prompt: Qwen step-by-step | Final |
| D5 | Can solve = 3 of 5; unsolvable left out | Final (in code) |
| D6 | Students Gemma-3 270M / 1B, full fine-tuning, cross-entropy on soft labels | Final (soft loss to implement) |
| D7 | AdamW, 3 epochs, batch 8, lr 2e-5; one-seed check first | Final |
| D8 | Latency is the primary measure | Final |
| D9 | Seeds: 5 main, 2 supporting | **Pending** |
| D10 | Large model on train (oracle baseline) | **After pilot** |

Not used, on purpose: **LoRA** (students are small), **DPO** (no preference pairs), **RL with the
oracle as reward** (the student would learn from the oracle, not the teacher; future work).

---

## 14. Changes from the submitted version

| Submitted | Now | Addresses |
|---|---|---|
| Candidates 270M / 0.5B / 1B | 1.5B / 14B | R1: router as big as candidates |
| Teacher Qwen2.5-3B, asked once (lost to always-large, 0.592 vs 0.651) | Qwen2.5-14B, 5 votes, checked before training | R1, R2: weak teacher |
| "Don't write it down" prompt, 1 greedy answer | Step-by-step prompt, 5 answers | Reasoning suppressed |
| String-match checking | math-verify (+46 / 3,000 correct, 0 lost) | R2 |
| Unsolvable labelled large (458 / 1,000 test) | Left out of routing accuracy | Inflated student gain |
| "small" undefined | Small = 1.5B | R2 |
| Hard labels | Soft labels (vote share) | – |
| 1,000 / ~748 / 500 data | 2,000 / 300 / 500, MATH-500, GSM-Symbolic | R2 |
| Teacher and fixed rules only | + random mix, TF-IDF, DeBERTa, oracle-trained student | R2 |
| Mean latency, HuggingFace teacher | p50/p95/p99, vLLM, overhead, pay-off | R2: 153× ignored costs |
| Seed bootstrap, 80-run correlation | Question + seed bootstrap | R1, R2 |
| Headline numbers only | + noisy teacher, size ladder, error analysis | R1, R2 |

Unchanged: problem, RQs, teacher → student distillation, the two student types, students, head.

---

## 15. Ideas for later (one at a time)

| Idea | Why it might help |
|---|---|
| Show the teacher more examples with known outcomes | Better teacher, better labels (P2) |
| Set the decision threshold from calibration (ECE) | Honest confidence makes the cost-quality curve reliable |
| Consistency cascade as an extra baseline | Compares two kinds of expensive router |

---

## Appendix: definitions

- p̂_m(x) = correct answers of model m on x ÷ 5; can solve if p̂ ≥ 0.6.
- Teacher soft label = share of 5 votes for "large"; hard label = majority.
- Student: large if P(large) ≥ 0.5.
- End-to-end accuracy = mean p̂ of the chosen model.
- Cost = 2 × parameters × (input + output tokens), router included.
- PGR = (acc − acc_small) / (acc_large − acc_small); APGR = area under PGR vs share sent to large
  (11 points; random = 0.5); CPT(x%) = smallest share sent to large with PGR ≥ x%.
- ECE = average gap between predicted confidence and actual accuracy, over confidence bins.
- Error bars: 1,000 bootstrap resamples of questions and seeds, 95%, Holm within each RQ.

---

## References ("bib" = in `research/references.bib`)

| Work | Used for | Source | Bib |
|---|---|---|---|
| Hinton et al., 2015 | Distillation, soft labels | – | ✅ |
| Stanton et al., NeurIPS 2021 | Fidelity vs correctness | arXiv:2106.05945 | add |
| RouteLLM | Pair choice; APGR, CPT | arXiv:2406.18665 | ✅ |
| RouterBench; LLMRouterBench | Offline evaluation with pre-computed answers | arXiv:2403.12031; 2601.07206 | add |
| Hybrid LLM | Pair choice | arXiv:2404.14618 | add |
| Router-R1; Zooter; RouterDC | Related routers | arXiv:2506.09033; bib; 2409.19886 | Router-R1, Zooter ✅ |
| Wu et al. 2026, *Student-Guided Teacher Distillation for Efficient LLM Task Routing* (preprint) | Closest related work; P1-P4, P6, P7. Differs: routes to 60 task categories (not models), NLI-classifier teacher (not an LLM router), graded only by teacher agreement (no outcome oracle) | arXiv:2610.02516 (read) | add |
| *When Routing Collapses* (2026) | P5 | arXiv:2602.03478 | add after reading |
| *Zero-Shot Confidence Estimation for Small LLMs* (2026) | P6 | arXiv:2605.02241 | add after reading |
| Yue et al., ICLR 2024 | Cascade (optional baseline) | arXiv:2310.03094 | add if used |
| Qwen2.5; Qwen2.5-Math card | Scores; prompt | qwenlm.github.io/blog/qwen2.5-llm; huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct | ✅ |
| GSM8K; MATH; MATH-500; GSM-Symbolic | Data | Hugging Face; arXiv:2410.05229 | GSM8K, MATH ✅ |
| Math-Verify | Answer checking | github.com/huggingface/Math-Verify | add |
| vLLM metrics; MLPerf | TTFT; p99 | docs.vllm.ai; arXiv:1911.02549 | read first |
