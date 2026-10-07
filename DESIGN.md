# DistillRouter: Research Design

---

## 0. Summary in one paragraph

Companies send each user question to one of several AI models: a cheap small model
for easy questions, an expensive large model for hard ones. The method that makes
this choice is called a **router**. The best routers are themselves large AI models,
so they are slow and add delay to every question. We take such a slow, smart router
(the **teacher**) and train a much smaller, faster model (the **student**) to copy its
decisions. This imitation is called **distillation**. We then check three things: how
well the student imitates the teacher, whether giving the student the teacher's
explanations helps, and whether better imitation leads to better routing.

---

## 1. The problem

> **Can the routing policy of a larger teacher model be distilled into a lightweight model?**

> *Given a teacher routing policy π_T, can we learn a lightweight routing policy π_θ
> that preserves routing accuracy while significantly reducing routing latency and
> computational overhead?*

**In plain words:** can a small model that learns from the teacher (the student) make
the same good routing decisions as the large teacher, much faster?

*Note: this wording differs slightly from `research.tex` ("an expensive model" →
"a larger teacher model"). The paper's problem statement must be updated to match (§16).*

---

## What we have implemented, and what we are improving

### A. What we have implemented

| Part | Implemented | Where |
|---|---|---|
| Candidate models | Gemma-3-270M-it (small), Qwen2.5-0.5B-Instruct (medium), Gemma-3-1B-it (large) | `common/config.py` (`DEFAULT_CANDIDATE_ROSTER`) |
| Candidate answers | One answer per model, greedy decoding, up to 1,024 new tokens | `common/hf_generation.py` (`do_sample=False`), `candidate/base.py` (`MAX_NEW_TOKENS`) |
| Answer checking | Extract the final answer (`\boxed{}`, JSON or fallback), then string match; no symbolic equivalence | `common/scoring.py`, `research.tex` §4.3 |
| True (oracle) labels | Cheapest model that answers correctly; if none does, labelled **large** | `oracle/labeler.py` (line 86) |
| Teacher | Qwen2.5-3B-Instruct (`qwen2.5-3b-v2`), two steps: zero-shot explanation, then few-shot label (2 examples per label from the calibration split); asked once, greedy | `teacher/qwen_teacher_v2.py`, `teacher/fewshot.py` |
| Label types | Binary (main) and 3-label | `research.tex` §4.2 |
| Data | Per dataset: train 1,000, validation 747 (GSM8K) / 749 (MATH), test 500, calibration 20 | `research.tex` Appendix Table (Dataset Splits), `common/config.py` |
| Student | Gemma-3-270M-it and Gemma-3-1B-it; last-token representation + small classification head | `student/train_classifier.py`, `research.tex` §3.4 |
| Training | Route-only: cross-entropy on the teacher's **hard** label. Reasoning-plus-route: + next-word loss on the teacher's explanation. AdamW, lr 2e-5, batch 8, 3 epochs; rarer label oversampled | `student/train_classifier.py` (line 108), `student/train_dual.py` (line 147), `student/data.py` (`oversample_minority_tiers`) |
| Checkpoint choice | Best validation macro-F1 against the teacher | `evaluation/router_eval.py` (`select_best_router_checkpoint`) |
| Repeats | 10 seeds per configuration | `multiseed/` |
| Measurement | Fidelity to teacher, oracle accuracy, end-to-end accuracy, router latency (HuggingFace generation for the teacher); paired bootstrap over seeds; correlation across seed runs | `evaluation/`, `research.tex` §5 |

**Results we reported (binary, 270M):**

| Router | Oracle accuracy | End-to-end accuracy | Router latency |
|---|---|---|---|
| Always large | 0.651 | 0.456 | – |
| Teacher | 0.592 | 0.392 | 1,633 ms |
| Route-only student | 0.682 ± 0.011 | 0.405 ± 0.005 | 10.9 ms |
| Reasoning-plus-route student | 0.692 ± 0.008 | 0.408 ± 0.005 | 11.2 ms |

**Problems we found after submission (checked on our own data):**
- 458 of 1,000 test questions were answered correctly by **no** candidate but were labelled
  "large". This raised every router's oracle accuracy, including the "always large" rule.
- On the 542 questions that some candidate does solve, oracle accuracy is
  teacher 0.624, route-only 0.626, reasoning-plus-route 0.627: the students' advantage
  over the teacher disappears.
- The end-to-end numbers above use a fixed rule (binary "small" → Qwen2.5-0.5B); we
  reproduced 0.405 and 0.408 exactly. The paper text did not state this rule.

### B. What we are improving

| # | Part | Implemented | Improvement | Reason |
|---|---|---|---|---|
| 1 | Label types | Binary + 3-label | **Binary only** | Focus; 3-label can be added later without redoing anything |
| 2 | Candidate models | 270M / 0.5B / 1B | **Qwen2.5-1.5B (small), Qwen2.5-14B (large)** | Reviewer 1: the router was as big as the models it chose between |
| 3 | Teacher | Qwen2.5-3B, asked once | **Qwen2.5-14B, asked 5 times** | Both reviewers: the teacher lost to "always large" |
| 4 | What the student learns | Hard label | **Soft label** (teacher vote share), hard label kept as comparison (E11) | Carries the teacher's uncertainty; tested, not assumed |
| 5 | Candidate answers | 1 per model | **5 per model**; "can solve" = correct 3 of 5 times | One answer can be right or wrong by luck |
| 6 | Answer checking | String match | **`math-verify`** (symbolic equivalence) | Reviewer 2 |
| 7 | Unsolvable questions | True label = large | **Teacher labels:** no change needed; the teacher sees only the question, so it always answers small or large and "unsolvable" never appears. **True labels:** a question no model solves has no correct routing decision, so it is **left out of routing accuracy** and **kept in end-to-end accuracy and cost** (§6.2) | The error found above; avoids inventing a label |
| 8 | Meaning of "small" and "large" | "small" could mean either of two models | Each label = **one fixed model**: small = Qwen2.5-1.5B, large = Qwen2.5-14B. Chosen by three rules from the literature (§5.1): a clear quality gap, a clear cost gap, and a small model that is good enough for many questions | Reviewer 2; RouteLLM and Hybrid LLM choose pairs this way |
| 9 | Data | 1,000 / ~748 / 500 per dataset; MATH test sampled | **2,000 / 300 / 500**; MATH test = MATH-500; extra test = GSM-Symbolic | More training data; standard test set; memorisation check |
| 10 | Generation | HuggingFace, one at a time | **vLLM** | Speed; fair teacher latency |
| 11 | Baselines | Teacher, always-small/medium/large | Five fair groups tied to our question (§9): fixed rules; **cheap routers without distillation** (the small 1.5B model prompted to route); cheap routers trained on the **same teacher labels** (TF-IDF, DeBERTa); the teacher itself; reference routers trained on true labels (not competitors) | Reviewer 2; each group answers one specific question |
| 12 | Measures | Accuracy, macro-F1, router latency (mean) | **Latency first** (§10): router latency p50/p95/p99, router overhead as a % of response time (TTFT and full response), throughput; then quality must be preserved (accuracy, cost-quality) | Our main claim is latency; measures follow vLLM's benchmark and routing papers |
| 13 | Statistics | Bootstrap over seeds; correlation across 80 runs | Fixed now: bootstrap over **test questions and seeds**; no correlation across runs. Details (Holm correction, E5 line fit) finalised **before** the test set is opened | Both reviewers; deciding after seeing results would weaken trust |
| 14 | Experiments | Main grid | + **E4** noisy teacher, **E5** copying vs routing, **E6** size ladder, **E7** explanation control, **E8** changed questions, **E9** teacher confidence, **E10** ablations, **E11** soft vs hard | Reviewer 1's main request; makes RQ2 and RQ3 testable |
| 15 | Process | Results produced, then text written | Checks before training (§11); rules committed before test results (§12); every number generated by scripts | Trustworthy results |
| 16 | Seeds | 10 per configuration | **5** for the main grid and E11; **2** for supporting experiments E4-E10 (D15) | Saves time; main claims keep enough runs to show variation |

**Unchanged:** RQ1-RQ3, the two training types, the student models, the
classification head, and the training settings. The problem statement keeps its
meaning; only its wording changes (§1).

---

## Final decisions (2026-10-06)

| # | Decision | Based on |
|---|---|---|
| D1 | Binary routing only (small vs large) | Our choice: simpler; 3-label later |
| D2 | Small = Qwen2.5-1.5B-Instruct, large = Qwen2.5-14B-Instruct (same family) | **Verified:** official Qwen2.5 scores (GSM8K 73.2 / 94.8, MATH 55.2 / 80.0). **Why one family:** in our old data, with a mixed pair (Qwen-0.5B vs Gemma-1B), the smaller model solved 276 of 3,000 questions the larger one failed, which blurs "small" vs "large". One family keeps the order cleaner (checked in S6) |
| D3 | Backup if 14B is too slow: large = Qwen2.5-7B-Instruct | **Verified:** official scores (GSM8K 91.6, MATH 75.5) |
| D4 | Teacher = Qwen2.5-14B-Instruct (the same model as the large candidate), 5 votes, majority | Our choice; quality **known only after** the teacher check. **Why the large model:** it is the strongest model we run, and it is exactly the kind of expensive router the paper is about: asking it costs about as much as one large-model call per question. **Known risk:** it may judge difficulty from its own ability rather than the 1.5B model's; the teacher check (§11) measures this. A teacher from another family is an optional ablation (E10) |
| D5 | "Can solve" = correct in at least 3 of 5 tries | Our choice (majority rule) |
| D6 | True labels: questions **no model solves** have no correct routing decision → excluded from routing accuracy; kept in end-to-end accuracy and cost. The soft information (p̂_small, p̂_large = share of 5 answers correct) is kept for every question and used directly for end-to-end accuracy | **Verified on our data:** the old rule mislabelled 458 of 1,000 test questions as large |
| D7 | Students = Gemma-3-270M-it and Gemma-3-1B-it | Unchanged from the paper |
| D8 | Keep E1-E9 and E11; E10 only if time allows | E4, E5 answer the reviewers' main points; E11 tests the soft-label improvement |
| D9 | Checks and thresholds exactly as in §11 | Our choice; written before any results |
| D10 | If the teacher check fails after 2 prompt fixes: try the 14B teacher with example questions that show which model solved them; if it still fails, report it honestly and keep E4-E7 | Our choice |
| D11 | Speed limit for all generation: 3 days | **Estimate;** the pilot (S4) gives the real number |
| D12 | Fairness rules in §12 apply without exception | Our choice |
| D13 | **Students learn soft labels** (the teacher's vote share, e.g. 4 of 5 said large → 0.8 large) instead of hard labels; hard labels kept as a comparison (E11) | Standard knowledge distillation (Hinton et al., 2015); same loss formula as the paper; **benefit known only after E11** |
| D14 | **Soft true labels:** for every question keep p̂_small and p̂_large (share of 5 answers correct). End-to-end accuracy uses them directly; a hard true label is used only for routing accuracy on solvable questions | Uses all the information from the 5 answers; no arbitrary label for unsolvable questions |
| D15 | **Pending (decide before training starts, S9).** Proposed: **5** seeds for the main grid and E11 (the claims we report); **2** for E4-E10 (supporting experiments). Alternative: 2 for everything | Our choice to save time. Two seeds cannot show how much results vary between runs, so they are not used for main claims; Reviewer 1 noted that two old runs differed as much as the reported effect |
| D16 | Latency is the primary measure (§10); quality must be preserved | Matches the problem statement |

**Not yet known (will be measured, not assumed):** how fast the 14B model runs on our
GPU; the real share of questions needing the large model (estimate ~23%); whether the
teacher beats random mixing.

---

**Scope of this version: binary routing only (small vs large).** 3-label routing is
left for later (§16); nothing in this plan has to be redone to add it.

---

## 2. Key words

| Word | Meaning |
|---|---|
| **Candidate models** | The models that actually answer questions: here one small and one large |
| **Router** | The program that picks which candidate answers each question |
| **Teacher** | The big, slow, smart router we want to imitate |
| **Student** | The small, fast router we train to imitate the teacher |
| **Distillation** | Training the student on the teacher's decisions |
| **Label** | The decision for one question: **small** or **large** |
| **True label (oracle label)** | The correct decision, found by actually running both candidates and checking whether the small one is good enough. Used only to grade, never to train our student |
| **Fidelity** | How often the student agrees with the teacher |
| **Routing accuracy** | How often the router makes the correct small/large decision |
| **End-to-end accuracy** | How many questions are finally answered correctly by the chosen model |
| **Router latency** | Time the router takes to decide which model answers |
| **Response latency** | Time until the user gets the answer: router latency + the chosen model's time. Measured as time to first token (TTFT) and time to the full answer |
| **Router overhead** | Router latency as a percentage of response latency: how much of the user's wait is spent on routing |
| **Seed** | One training run with a different random start. 5 seeds for main results, 2 for supporting experiments (D15) |

---

## 3. Research questions (from `research.tex`, unchanged)

| RQ | Question (paper wording) | In plain words |
|---|---|---|
| **RQ1** | How effectively can a routing policy be transferred from an accurate but expensive teacher router to a lightweight distilled router? | How well does the student copy the teacher, and how much faster is it? |
| **RQ2** | Does augmenting routing supervision with teacher reasoning improve policy transfer relative to route-only distillation? | Does showing the student the teacher's explanations, not just its answers, help it copy better? |
| **RQ3** | Do improvements in policy transfer translate into improvements in routing accuracy under independent oracle labels? | If the student copies the teacher better, does it also route better? |

---

## 4. What we keep and what we change

| Part | Submitted paper | Revision | Why |
|---|---|---|---|
| Problem, RQs, method | – | **Unchanged** | – |
| Label types | Binary (main) + 3-label | **Binary only for now** | Focus; 3-label later |
| Student sizes | 270M and 1B (Gemma-3) | **Unchanged** | – |
| Two training types | Route-only, reasoning-plus-route | **Unchanged** | – |
| Candidate models | 3 models: 270M / 0.5B / 1B | **2 models: Qwen2.5-1.5B (small), Qwen2.5-14B (large)** | Reviewer 1: the router was as big as the models it chose between. Binary needs only two |
| Teacher | Qwen2.5-3B, asked once | **Qwen2.5-14B, asked 5 times, majority vote** | Both reviewers: the teacher lost to "always use large" |
| What the student learns | Hard label ("large") | **Soft label** (vote share, e.g. 0.8 large / 0.2 small) | Carries the teacher's uncertainty; unsure questions count less (§7) |
| Answer checking | Exact text match | **`math-verify`** (accepts "1/2" = "0.5") | Reviewer 2 |
| Each candidate answers | Once | **5 times** | One answer can be right or wrong by luck |
| Questions neither model can answer | Labelled "large" | **Flagged; left out of routing accuracy; kept in end-to-end accuracy and cost** | This made our old results look better than they were |
| Meaning of "small" | Unclear (R2) | **Always the 1.5B model** | Reviewer 2 |
| Comparisons | Teacher, fixed rules | **More baselines** (§9) | Reviewer 2 |
| Error bars | Training repeats only | **Also resample test questions** | Both reviewers |
| New experiments | – | **Noisy-teacher test, copying-vs-routing test, size ladder, explanation control** (§8) | Reviewer 1's main point |

---

## 5. Setup

### 5.1 Models

| Role | Model | Notes |
|---|---|---|
| Small candidate | Qwen2.5-1.5B-Instruct | Cheap; handles many easy questions |
| Large candidate | Qwen2.5-14B-Instruct | About 9× the small one; about 50× the 270M student |
| Teacher | Qwen2.5-14B-Instruct | Two steps: write a short explanation, then answer "small" or "large" |
| Student | Gemma-3-270M-it and Gemma-3-1B-it | Same starting point for every variant |

**How "small" and "large" are chosen (from the literature):**

| Paper | Small (weak) | Large (strong) | How they chose |
|---|---|---|---|
| RouteLLM | Mixtral-8x7B | GPT-4 | Strong from the top two quality tiers of Chatbot Arena, weak from the third tier; over 100× price difference. Also tested Claude 3 Sonnet / Opus and Llama 3.1 8B / 70B |
| Hybrid LLM | e.g. Llama-2-7B, FLAN-T5 (800M) | e.g. Llama-2-13B, GPT-3.5-turbo | Pairs with small, medium and large quality gaps; notes that models of the same architecture have a small gap; small = fits on an edge device, large = cloud |

**Our rules**, taken from these papers:
1. A **clear quality gap** (otherwise routing cannot gain anything).
2. A **clear cost gap** (otherwise routing cannot save anything).
3. The small model is **good enough for many questions** (otherwise everything goes to large).

**Why 1.5B and 14B:**

| | GSM8K | MATH |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 73.2% | 55.2% |
| Qwen2.5-14B-Instruct | 94.8% | 80.0% |
| Expected share needing large | ~22% | ~25% |
| Expected share unsolvable | ~5% | ~20% |

*(Accuracies from the official Qwen2.5 release; expected shares are rough estimates.)*

- Rule 1: quality gap of about 22 points (GSM8K) and 25 points (MATH).
- Rule 2: about 9× more parameters (1.5B vs 14B), so roughly 9× more compute per answer.
- Rule 3: the small model alone solves about 73% of GSM8K and 55% of MATH.
- Same family (Qwen2.5): Hybrid LLM observes smaller gaps within one architecture; here
  the gap is still large because the sizes differ by about 9×. One family also keeps
  "larger is better" more consistent (D2).
- About **1 in 4** questions should need the large model. That is a realistic mix
  for routing, and our measures (macro-F1, oversampling of the rarer label, cost-quality curves) handle it.
- The large model is about **9× the compute** of the small one, so routing saves real money.
- The router reads the question once; the small model reads it **and** writes
  hundreds of words. So the 270M student costs about 1/15 of the small model's
  answer, and the 1B student about 1/4. The router is never the expensive part
  (Reviewer 1's concern).

We do **not** use a 0.5B small model to get a 50/50 split: the 1B student would then
cost about as much as the small model's answer, which is the problem Reviewer 1 raised.

**Backup plan:** if the speed test (§11) shows 14B is too slow on our GPU, use
Qwen2.5-1.5B (small) and Qwen2.5-7B (large), with the 7B as teacher (expected
share needing large: ~19% GSM8K, ~20% MATH). Nothing else changes.

All large models run with **vLLM** (fast serving software), which also makes the
teacher's speed measurement fair.

### 5.2 Data

| Part | GSM8K | MATH | Used for |
|---|---|---|---|
| Training | 2,000 | 2,000 | Teacher labels these; the student learns from them |
| Validation | 300 | 300 | Choosing the best training checkpoint; checks in §11 |
| Examples | 20 | 20 | Example questions shown to the teacher in its prompt |
| Test | 500 | 500 (MATH-500) | Final results. Used once, never for tuning |
| Extra test (new) | 500 questions from GSM-Symbolic (`apple/GSM-Symbolic` on Hugging Face: GSM8K questions with changed names and numbers) | – | Checks that results hold on questions the models have not memorised |

No question appears in two parts.

---

## 6. How we make the labels

### 6.1 Running the candidates
Both candidates answer every question **5 times** (same prompt for both; final answer
in `\boxed{}`; temperature 0.7; up to 1,024 words of output). For each model and
question we record how many of the 5 answers were correct, plus the number of words
read and written and the time taken.

A model **can solve** a question if it is correct **at least 3 times out of 5**.

### 6.2 True labels (for grading only)

For every question we keep the **soft true label**: p̂_small and p̂_large, the share of
the 5 answers each model got right (e.g. 0.4 and 1.0). From it we derive a hard label:

| Small model can solve? | Large model can solve? | Hard true label | Used in routing accuracy? |
|---|---|---|---|
| Yes | – | **small** | Yes |
| No | Yes | **large** | Yes |
| No | No | none, flagged *unsolvable* | **No**: no routing decision is correct, so it cannot be scored. It still counts in end-to-end accuracy and cost |

- **End-to-end accuracy** uses the soft label directly: a router that picks large on
  a question with p̂_large = 0.8 earns 0.8 for that question.
- **Routing accuracy** uses the hard label, on solvable questions only.
- We also report the share of unsolvable questions and how often the small model solves
  a question the large one does not.

**Note:** "unsolvable" exists only for true labels. Teacher labels never have it: the
teacher sees only the question and always answers small or large.

### 6.3 Teacher labels (what the student learns from)
1. The teacher writes a short explanation of how hard the question is (≤150 words).
2. The teacher answers **small** (a ~1-2B model is enough) or **large** (a ~14B model
   is needed). The options are described by size, not by product name.
3. We repeat this **5 times** (temperature 0.7) and take the **majority vote** (with 5
   votes there are no ties). For the "teacher asked once" baseline, we ask once with
   temperature 0, which is how the teacher would be used in practice.
4. We keep one explanation that agrees with the majority (used for reasoning-plus-route).
5. If the teacher's answer cannot be read, we ask again twice, then default to "large".

We also record **how much the 5 votes agree** (5/5, 4/5 or 3/5). This is the
teacher's confidence, used in E9.

**Soft label.** From the 5 votes we keep the share for each label:

| Votes for large | Soft label (small, large) | Hard label |
|---|---|---|
| 0 of 5 | (1.0, 0.0) | small |
| 1 of 5 | (0.8, 0.2) | small |
| 2 of 5 | (0.6, 0.4) | small |
| 3 of 5 | (0.4, 0.6) | large |
| 4 of 5 | (0.2, 0.8) | large |
| 5 of 5 | (0.0, 1.0) | large |

The soft label is what the student learns from (§7). The hard label (majority) is used
for checkpoint choice, for measuring fidelity, and for the hard-label comparison (E11).

---

## 7. How we train the student (method from the paper, with soft labels)

The student reads the question and outputs **small** or **large** in one quick step.

- **Route-only:** the student learns only the teacher's labels.
- **Reasoning-plus-route:** the student also learns to write the teacher's explanation.
  The explanation is used **only during training**. When deployed, both versions
  are equally fast.

**Soft labels (improvement).** The paper's route loss stays exactly the same formula:

  L_route = − Σ_c y_c · log ŷ_c

The only change is what y_c holds:

| | y for a question where 4 of 5 teacher votes said large |
|---|---|
| Submitted paper (hard) | (small 0, large 1) |
| Revision (soft) | (small 0.2, large 0.8) |

Why this should help:
- **More information per question.** The student learns not only *what* the teacher
  decided but *how sure* it was.
- **Less damage from teacher mistakes.** When the teacher is unsure (3 of 5), the
  student is pushed only gently, so a likely-wrong label teaches it less.
- **Better-calibrated probabilities.** The student's "large" probability becomes more
  meaningful, which makes the cost-quality curves (§10) more reliable.
- **Free.** The 5 votes are already collected for the majority label; no extra GPU time.

Whether it actually helps is **tested, not assumed** (E11).

Settings as in the paper (AdamW, learning rate 2e-5, batch 8, 3 epochs). The rarer
label is oversampled, as in the existing code (`oversample_minority_tiers`, grouped by the
hard label). The best checkpoint is chosen by agreement with the teacher's **hard**
label on validation (true labels are never used for training or choosing).

The student also outputs a probability for "large". Moving the cut-off (e.g. "large
if probability > 0.3" instead of > 0.5) trades cost for quality; we use this only to
draw cost-quality curves (§10). The deployed decision uses the normal 0.5 cut-off.

---

## 8. Experiments

### Main grid (binary routing)
**2 student sizes (270M, 1B) × 2 training types (route-only, reasoning-plus-route) × 5 seeds = 20 runs.**

| ID | Experiment | What it shows | RQ |
|---|---|---|---|
| **E1** | Fidelity and speed for every grid cell | How well and how fast the student copies the teacher | RQ1 |
| **E2** | Route-only vs reasoning-plus-route | Whether explanations help | RQ2 |
| **E3** | Routing accuracy, end-to-end accuracy, cost-quality curves vs teacher and baselines | Whether copying well means routing well. **Reported first in the results**, not set aside | RQ3 |

### New experiments (cheap: only the student is retrained, no new large-model runs except E8)

| ID | Experiment | What it shows | Answers |
|---|---|---|---|
| **E4 Noisy teacher** | Train on teacher labels damaged two ways: **(a) random** (flip 10/20/30% of labels at random); **(b) systematic** (e.g. "every MATH question → large", "every long question → large") | Whether the student ignores random mistakes but copies systematic ones. Explains why a student can disagree with its teacher in useful ways | Reviewer 1's main request |
| **E5 Copying vs routing** | Train with 250 / 500 / 1,000 / 2,000 / 4,000 teacher labels, plus E4's damaged labels | Many different levels of copying quality, so we can test properly whether better copying leads to better routing | RQ3; Reviewer 2's statistics point |
| **E6 Size ladder** | Same teacher labels, students of different sizes: simple word-count model (TF-IDF) → DeBERTa-v3-small → Gemma-270M → Gemma-1B | How small the student can be before quality drops. If the simplest one is enough, that is a useful finding | Reviewer 2's "simple classifier" request |
| **E7 Explanation control** | Reasoning-plus-route, but each question gets a **random other question's** explanation | Whether any gain comes from the explanation's content or just from extra training text | Makes RQ2 trustworthy |
| **E8 Changed questions** | Run all routers on the extra test set (§5.2) | Results do not depend on memorised benchmark questions | Data contamination; narrow domain |
| **E9 Teacher confidence** | Student-teacher agreement split by how much the teacher's 5 votes agreed | Whether the student mostly disagrees where the teacher itself was unsure | Reviewer 1's "noisy labeller" explanation |
| **E10 Ablations** | Teacher asked once vs 5 times; base vs instruct starting model; optional: a teacher from another family (e.g. Gemma-3-12B-it) to check that the teacher being the large model does not bias the labels | Whether these choices matter | D4's known risk |
| **E11 Soft vs hard labels** | Same students trained on the hard (majority) label instead of the soft label | Whether soft labels improve copying and routing | Supports the soft-label improvement (D13) |

**Noise with soft labels (E4).** A random "flip" mirrors the soft label (0.8 large
becomes 0.2 large). A systematic bias sets the label to (0, 1), i.e. certainly large.

E4-E7 and E9-E10 reuse labels we already have. E5 also tells us **how many teacher
labels are needed** (a practical cost question).

**Run sizes for the new experiments** (to keep them affordable):

| Experiment | Student | Conditions | Seeds | Runs |
|---|---|---|---|---|
| E4 | 270M, route-only | 3 random + 2 systematic | 2 | 10 |
| E5 | 270M and 1B, route-only | 5 training sizes | 2 | 20 |
| E6 | TF-IDF, DeBERTa-v3-small (route-only) | 2 new students | 2 | 4 |
| E7 | 270M and 1B, reasoning-plus-route | shuffled explanations | 2 | 4 |
| E10 | 270M | once vs 5 votes; base start | 2 | 4 |
| E11 | 270M and 1B, route-only | hard labels | **5** | 10 |

E11 uses 5 seeds (not 2) because soft labels are a claimed improvement and need the
same strength of evidence as the main grid (D15). With 2 seeds, results of E4-E10 are
reported as supporting evidence, with error bars from resampling test questions.

**How E5 is analysed:** for each condition we first average over its seeds, giving
one point per condition (copying score, routing score). We then draw the line through
these points and give an error bar on its slope. Seeds of one condition are never
counted as separate data points (this was Reviewer 2's objection).

---

## 9. What we compare against (baselines)

Each group answers one question about our research problem. All baselines use the same
questions, the same validation split for any tuning, and the same latency protocol (§10).

| Group | Baseline | What it is | Question it answers |
|---|---|---|---|
| **1. Fixed rules** (no learning) | Always small / always large | Same model for every question | Is routing worth it at all? |
| | Random mix | Random small/large at the same share of large | Is the router better than chance? |
| | Length rule, dataset rule | Long questions → large; MATH → large | Is the student just counting words or spotting the dataset? |
| **2. Cheap router without distillation** | Prompted small model | Qwen2.5-1.5B given the teacher's prompt, asked once | **Is distillation needed, or is prompting a small model enough?** (directly about our method) |
| **3. Same teacher labels, other students** | TF-IDF + logistic regression; DeBERTa-v3-small | Trained on the same teacher labels as our student (E6) | Does the student's size or type matter? |
| **4. The teacher** | Teacher asked once (temperature 0) / 5 times | The router we distil | How much quality do we keep, and how much latency do we remove? |
| **5. Reference only** (uses true labels, which our method never sees) | Student on true labels; BERT on true labels (RouteLLM-style); optional RouteLLM matrix factorization | Trained on true labels | How far is distillation from a router that had the true answers? **Not a competitor** |

**Fairness rules for baselines:** same training questions; same validation-only tuning;
latency measured the same way for all routers; group 5 is always marked as using extra
information.

---

## 10. How we measure

Latency is the main measure (D16). Quality must be **preserved**, so it is measured too.

### 10.1 Latency (primary)

| Measure | Plain meaning | Why |
|---|---|---|
| **Router latency, p50 / p95 / p99** | Typical, slow and worst-case time to decide | A mean hides slow cases; MLPerf uses the 99th percentile as its latency limit |
| **Router overhead (%)** = router latency ÷ response latency | Share of the user's wait spent on routing | Routing papers report overhead this way (DiSRouter, ParetoBandit); a 1.6 s router is small next to a 30 s answer, large next to a 1 s answer |
| **Added time to first token** | Extra wait before the first word appears | TTFT is the responsiveness users notice (vLLM benchmark metrics; latency-aware routing work) |
| **Response latency, p50 / p95** for the full system | Router + chosen model, per question | Compared with always-small and always-large |
| **Router throughput** | Routing decisions per second at batch sizes 1, 8 and 32 | Matters when many users arrive at once |
| **Speedup** = teacher latency ÷ student latency | How much faster than the teacher | The old "153×", now measured fairly |
| **Setup cost and pay-off** | GPU-hours to label with the teacher and train the student; number of questions after which it pays off | Reviewer 2: the 153× ignored these costs |

**Fair measurement protocol:**
- Same GPU for all routers; nothing else running.
- Teacher served with **vLLM** (not slow HuggingFace generation).
- Clock runs from receiving the question text to having a decision, **including tokenisation**;
  model loading excluded; GPU synchronised before stopping the clock.
- 20 warm-up questions, then all 1,000 test questions, one at a time.
- Students also measured on **CPU** (one thread and all cores).
- Candidate response times (TTFT and full answer) taken from the vLLM runs in S5,
  measured one question at a time on a 100-question sample.

### 10.2 Quality (must be preserved)

| Measure | Plain meaning |
|---|---|
| Fidelity: accuracy and macro-F1 against the teacher | How often the student agrees with the teacher (macro-F1 counts the rarer label fairly) |
| Routing accuracy, **solvable questions only** | How often it makes the right small/large call when a right call exists |
| End-to-end accuracy | How many questions end up answered correctly, using the soft true labels (D14) |
| Compute cost | 2 × parameters × tokens, for the chosen model + the router |
| Cost-quality curve; APGR; CPT(50%), CPT(80%) (RouteLLM) | Whether the router gives more quality for the cost than random mixing |
| Cost saved at 95% of always-large quality | One easy headline number |

---

## 11. Checks before training (stop and fix if any fails)

All checks run on the **validation** questions first, so we do not waste GPU time.

| Check | Pass if | If it fails |
|---|---|---|
| Speed test | 50 questions per model predicts all generation finishes in ≤ 3 days | Use the backup models (§5.1) |
| Answer checker | `math-verify` agrees with a human on ≥ 95 of 100 answers | Fix the answer format |
| Solvable | Fewer than 20% of questions unsolvable | Change models |
| Enough "large" | At least 15% of questions need the large model (expected ~23%) | Use a larger "large" model, or drop the easiest questions from training only |
| **Teacher is good** | Teacher beats random mixing of small and large at the same cost | Show the teacher example questions with known results; at most 2 prompt changes; then a bigger teacher |
| No overlap | No question in two parts | Rebuild the splits |

---

## 12. Fairness rules

- **Test once.** Nothing is tuned on the test set.
- **Write rules first.** This document, with its thresholds, is committed to git
  before we run the test set, so results cannot shape the rules.
- **Two kinds of error bars.** We resample both training repeats and test
  questions. This shows whether a result would hold on different questions.
- **Correct for many tests.** When we test the same question in several settings,
  we adjust (Holm correction) so a lucky result is not reported as real.
- **Timing of statistical choices.** The two rules above are fixed now. Any remaining
  details are written into this document **before** the test set is opened (S12),
  never after seeing test results.
- **Honest wording.** If error bars include zero, we say "no difference found", not "the same".
- **Per dataset.** Results are shown for GSM8K and MATH separately as well as together.

---

## 13. Risks and plans

| Risk | Plan |
|---|---|
| Even the 14B teacher is not better than random mixing | Give it example questions with known results. If it still fails, report it honestly. E4-E7 still hold, because they study copying, not teacher quality |
| Too few questions need the large model (below 15%) | Use a larger "large" model; do not shrink the small model, which would make the router as costly as the small model again |
| The simple TF-IDF student is as good as Gemma | Report it as a finding (E6): distillation works even into a tiny model |
| Explanations still do not help | E7 makes a "no effect" result clear and useful |
| Large models are too slow on our GPU | Backup models (§5.1) |

---

## 14. Order of work

| Step | Task | Needs big GPU runs? |
|---|---|---|
| 1 | Install vLLM and `math-verify`; update the model list in `common/config.py` | No |
| 2 | Speed test + answer checker test | Small |
| 3 | Both candidates and teacher on validation → checks in §11 | Yes |
| 4 | Both candidates and teacher on training and test (+ extra test) | Yes |
| 5 | Train main grid (E1-E3) and baselines | No |
| 6 | New experiments E4-E7, E9, E10, E11 | No |
| 7 | Speed and cost study; E8 | Small |
| 8 | Update the paper | No |

Dropping the medium model cuts the candidate GPU time by about a third.
The full step-by-step pipeline, with every input, output and check, is in **§18**.

**Can start now:** E4 (noisy teacher) and E6 (size ladder) can be tried on our
**current binary** data as a pilot, before the new models are ready.

---

## 15. Changes to execute

Every change, in the order to do it. "Done when" is the test that proves the change works.
Old code paths stay available behind settings so the submitted results can still be reproduced.

### Phase A: Environment and data (no GPU)

| # | Change | Files | Done when |
|---|---|---|---|
| A1 | Add `vllm` and `math-verify` | `requirements.txt` | Both import; vLLM generates one answer on the GPU |
| A2 | New settings: small = Qwen2.5-1.5B-Instruct, large = Qwen2.5-14B-Instruct, teacher = Qwen2.5-14B-Instruct, answers per question = 5, temperature 0.7, "can solve" = 0.6, binary label space by default; old roster kept under a separate name | `common/config.py` | Old and new settings both load |
| A3 | Add MATH-500 (MATH test) and GSM-Symbolic (extra test) loaders | `dataset/` (new `math500.py`, `gsm_symbolic.py`), `dataset/__init__.py` | Both write `data/processed/.../*.jsonl` in the `Example` format |
| A4 | New split sizes (2,000 / 300 / 20 / 500); remove MATH-500 questions from the MATH training pool; stratify MATH by subject and level | `dataset/base.py`, `dataset/math.py` | `check-splits` reports zero overlap |
| A5 | Write the sealed split manifest (IDs + checksums) | `common/split_integrity.py`, `run.py` | `split_manifest.json` exists and is committed |

### Phase B: Answer checking and candidates (GPU)

| # | Change | Files | Done when |
|---|---|---|---|
| B1 | Use `math-verify` for GSM8K and MATH scoring; keep the old scorer selectable | `common/scoring.py`, `dataset/gsm8k.py`, `dataset/math.py` | Unit test: "1/2" = "0.5", "\frac{1}{2}" = "0.5", wrong answers rejected |
| B2 | vLLM backend: batched generation, 5 samples, temperature 0.7, top-p 0.95, 1,024-token limit; HuggingFace path kept for latency comparison | `common/hf_generation.py` (or new `common/vllm_generation.py`), `candidate/base.py` | 50 questions answered; speed recorded |
| B3 | Answer records gain: sample number, extracted answer, `truncated` | `common/schema.py` (`CandidateAttempt`) | Records written and resumed after an interrupted run |
| B4 | Pilot command: speed per model + 100 answers exported for hand-checking | `run.py pilot` (new) | Report written to `results/pilot/` |

### Phase C: Labels (GPU for the teacher)

| # | Change | Files | Done when |
|---|---|---|---|
| C1 | True labels: soft label (p̂_small, p̂_large), 3-of-5 rule, hard label from §6.2, unsolvable flag (no hard label) | `oracle/labeler.py`, `common/schema.py` (`OracleLabel`) | Label report shows small / large / unsolvable shares |
| C2 | New binary teacher (`qwen2.5-14b-v3`): size-based prompt, explanation then label, 5 votes at temperature 0.7 + 1 at temperature 0, retry rule | `teacher/` (new `qwen_teacher_v3.py`), `teacher/fewshot.py`, `teacher/labeler.py` | Validation labelled; parse-failure rate reported |
| C3 | Teacher records gain: the 5 votes, **soft label**, majority label, agreement, temperature-0 label | `common/schema.py` (`TeacherLabel`) | Soft labels sum to 1 for every question |
| C4 | Teacher check: teacher vs random mixing on validation | `evaluation/oracle_check.py` | Pass/fail printed; pipeline stops on fail |

### Phase D: Student (no large GPU jobs)

| # | Change | Files | Done when |
|---|---|---|---|
| D1 | Training files carry the soft label alongside the hard label | `student/data.py` | Leak check: no true-label fields in any student file |
| D2 | **Soft-label loss:** pass the soft label as the target of `F.cross_entropy` (PyTorch accepts probability targets); setting `--label-type soft|hard` | `student/train_classifier.py` (line 108), `student/train_dual.py` (line 147) | Soft = hard when all votes agree (5/5 or 0/5); unit test passes |
| D3 | Oversampling groups by the hard label | `student/data.py` (`oversample_minority_tiers`) | Rare label share rises as configured |
| D4 | Variants for E4, E5, E7, E10, E11: noise (mirror / systematic), training-size subsets, shuffled explanations, teacher-once, hard labels; all with fixed seeds | `student/data.py`, `run.py build-binary-student-data` | Each variant file written with its seed in `manifest.json` |
| D5 | Predictions file saves the "large" probability for validation, test and GSM-Symbolic | `evaluation/router_eval.py` | `predictions.jsonl` has probabilities for all three |

### Phase E: Baselines, speed, evaluation (no large GPU jobs)

| # | Change | Files | Done when |
|---|---|---|---|
| E1 | Fixed rules, random mixing, length rule, dataset rule | new `evaluation/baselines.py`, `run.py baselines` | Same `predictions.jsonl` format as students |
| E2 | Prompted small model (Qwen2.5-1.5B with the teacher's prompt); TF-IDF + logistic regression and DeBERTa-v3-small on teacher labels; BERT and student on true labels (reference) | new `evaluation/baselines.py` | Trained or run, saved like S9 |
| E3 | Speed study (§10.1): one question at a time, warm-up, p50/p95/p99, overhead %, added TTFT, throughput at batch 1/8/32, GPU and CPU, teacher on vLLM | new `evaluation/latency.py`, `run.py latency` | `results/latency/` written |
| E4 | Measures: end-to-end accuracy, cost, cost-quality curve, APGR, CPT, solvable-only scores | `evaluation/metrics.py` | Unit test: random mixing gives APGR ≈ 0.5 |
| E5 | Error bars (bootstrap over questions and repeats), Holm correction, E5 line fit | new `evaluation/statistics.py` | Reproduces a known result on a toy example |
| E6 | All tables and figures generated from results | `research/figures/make_figures.py`, new table script | No hand-typed numbers |
| E7 | Makefile targets for the new pipeline | `Makefile`, `PIPELINE.md` | `make pilot`, `make labels`, `make train-v2`, `make evaluate-v2` run end to end |

---

## 16. What changes in the paper text

The RQs are **not** changed. The problem statement keeps its meaning but is reworded (§1).

| Section | Change |
|---|---|
| Introduction and Problem Formulation | Use the reworded problem statement from §1 |
| Problem Formulation | Candidate set for this version: two models (small, large) instead of three |
| Binary Routing | "small" = the small model, "large" = the large model; no more "either m₁ or m₂" |
| Three-Label Routing | Move to future work for now |
| Oracle Labels | Add the 5-answer, 3-of-5 rule, soft true labels, and the rule that unsolvable questions are left out of routing accuracy |
| Route-Only Distillation (Eq. for L_route) | Formula unchanged; state that y_c is the teacher's vote share (soft label), with hard labels as the comparison (E11) |
| Teacher Router | Teacher sampled 5 times; vote share and majority label both kept |
| Routing Environment | New models, teacher, answer checker, settings |
| Metrics | Latency first (p50/p95/p99, overhead %, added TTFT, throughput); then end-to-end, cost-quality, solvable-only, error-bar method |
| Results | Binary only; lead with end-to-end and cost-quality; add E4-E9 and E11 |
| Abstract, Conclusion | New numbers; drop "better than the teacher" unless the new results support it; drop 3-label claims |
| Related work | Group routers by where their training signal comes from: real outcomes (RouterDC), a scoring model (Zooter, BEST-Route, Hybrid LLM), trial and error (Router-R1), **an existing router's decisions (ours)** |

**Adding 3-label later:** run a medium model (e.g. Qwen2.5-7B) on the same questions
and ask the teacher for three labels. All binary results and the small/large runs are reused.

---

## 17. How each review point is answered

| Reviewer point | Where |
|---|---|
| R1: router as big as the candidates; no cost difference | §5.1 new models |
| R1, R2: teacher loses to "always large" | §5.1 stronger teacher; §11 teacher check |
| R1: student beats teacher only because it averages out noise; the deciding experiment was not run | E4, E9 |
| R1: end-to-end result set aside | E3 reported first |
| R1, R2: error bars only from training repeats | §12 |
| R2: "small" does not say which model runs | §4, §6.2: small = the 1.5B model |
| R2: missing split sizes, labelling details, exact-match checking | §5.2, §6, `math-verify` |
| R2: 153× ignores other costs | §10 setup cost and pay-off point |
| R2: no simple or published baselines | §9, E6 |
| R2: correlation over 80 runs is not valid | E5 + §12 |

---

## 18. The pipeline, step by step

This section describes the system that produces every number in the paper. It is
written so that someone new can run it from start to finish.

### 18.1 Design rules

| Rule | What it means in practice |
|---|---|
| **One stage, one job** | Each stage reads the files of the stage before and writes its own files. No stage edits another stage's files |
| **Files are the interface** | Every stage's output is a `.jsonl` file (one JSON record per line) in a fixed folder. Any stage can be re-run alone |
| **Resumable** | Every model answer is saved as soon as it is made, keyed by (question, model, sample number, prompt version). A crash loses nothing; re-running skips finished work |
| **Versioned** | Every output folder records the settings and code version that made it (`manifest.json`). Changing a prompt creates a new version folder; old results are never overwritten |
| **Checked** | Every stage ends with an automatic report. The pipeline stops if a check fails |
| **Test is sealed** | The test question IDs are fixed and their checksum is written down before any model is run. Nothing reads test results until step S12 |
| **No leaks** | True labels are never written into student training files. The build step checks this and stops if it finds one |
| **Same seed, same result** | All random choices (sampling, splitting, noise, shuffling) use recorded seeds |

### 18.2 Overview

```
S0  Environment          ─► working vLLM, math-verify, pinned versions
S1  Download             ─► data/raw/
S2  Clean                ─► data/processed/<dataset>/all.jsonl
S3  Split and seal       ─► data/processed/<dataset>/{train,validation,calibration,test}.jsonl
S4  Pilot                ─► speed + checker reports           (checks: speed, answer checker)
S5  Candidates answer    ─► data/oracle/<dataset>/<split>.attempts.jsonl
S6  True labels          ─► data/oracle/<dataset>/<split>.labels.jsonl   (checks: solvable, enough "large")
S7  Teacher labels       ─► data/teacher/<dataset>/<version>/<split>.jsonl (check: teacher is good)
S8  Student data         ─► data/student/<dataset>/<version>/<variant>/train.jsonl
S9  Train students       ─► runs/<experiment>/<variant>/seed_<n>/
S10 Baselines            ─► runs/baselines/
S11 Speed study          ─► results/latency/
S12 Evaluate             ─► results/tables/, results/figures/
S13 Update the paper     ─► research/research.tex
```

Validation always goes first through S5-S7, so the checks run before the large
GPU jobs on training and test.

### 18.3 The steps

**S0. Environment**
- Install: Python packages from `requirements.txt`, `math-verify`, and the NVIDIA vLLM container for DGX Spark.
- Record: GPU, driver, CUDA, vLLM, PyTorch and Transformers versions in `results/environment.json`.
- Check: load each model once and generate one answer.

**S1. Download** *(existing: `run.py prepare-data`)*

| Source | Hugging Face name | Used for |
|---|---|---|
| GSM8K | `openai/gsm8k` (config `main`) | train, validation, calibration, test |
| MATH | `EleutherAI/hendrycks_math` (7 subject configs; already used by `dataset/math.py`) | train, validation, calibration |
| MATH-500 | `HuggingFaceH4/MATH-500` | MATH test |
| GSM-Symbolic | `apple/GSM-Symbolic` (config `main`) | extra test (E8) |

Save the raw files and their checksums in `data/raw/`.

**S2. Clean** *(existing: `run.py prepare-data`, extended)*
1. Convert every question to the common format (`Example` in `common/schema.py`): ID, dataset, question, reference answer, subject and level (MATH).
2. Reference answers: GSM8K = the number after `####`; MATH = the content of the last `\boxed{}` in the solution; MATH-500 and GSM-Symbolic = their answer field.
3. Drop a question if `math-verify` cannot read its reference answer. Count and report how many.
4. Remove exact duplicates and remove every MATH-500 question from the MATH training pool (matched on normalised question text).

**S3. Split and seal** *(existing: `run.py prepare-data`, `run.py check-splits`)*
1. Draw the splits in §5.2 with a fixed seed. For MATH, keep the same mix of subjects and levels in every split.
2. Calibration (20 per dataset) is taken from the training pool before training is sampled.
3. Check: no question in two splits (exact text and normalised text).
4. Write `data/processed/split_manifest.json` with the question IDs and a checksum per split. Commit it to git. **From now on the test split is sealed.**

**S4. Pilot** *(new: `run.py pilot`)*
1. Run both candidates and the teacher on 50 validation questions.
2. Measure questions per hour for each model; project the total time for S5 and S7. → **Speed check.**
3. Hand-grade 100 answers (50 per dataset, mixed correct/incorrect according to `math-verify`); save to `results/audit/checker_audit.jsonl`. → **Answer-checker check.**

**S5. Candidates answer** *(existing: `run.py oracle-label`, changed to vLLM and 5 samples)*
- Order: validation → test → training → calibration → GSM-Symbolic.
- For each question and each candidate: 5 answers with the settings in §6.1.
- One record per answer: question ID, model, sample number (0-4), prompt version, answer text, extracted answer, correct (yes/no), tokens read, tokens written, time, and `truncated` (yes if the 1,024-token limit was hit).
- Saved immediately after each batch, so a crash loses at most one batch.
- Report: accuracy per model and dataset (should be close to the official numbers in §5.1), truncation rate.

**S6. True labels** *(existing: `oracle/labeler.py`, rewritten)*
- For each question: p̂_small, p̂_large, true label (table in §6.2), unsolvable flag.
- One record per question: question ID, p_small, p_large, label, unsolvable, plus the average tokens and time for each model (needed for cost).
- Report per dataset and split: share of small / large / unsolvable; how often the small model solves a question the large one does not. → **Solvable and enough-"large" checks** (on validation).

**S7. Teacher labels** *(existing: `run.py label-data`, changed to vLLM and 5 votes)*
1. Freeze the prompt; give it a version name (e.g. `t14b-v1`).
2. Validation first: 5 votes (temperature 0.7) and 1 answer at temperature 0 per question.
3. One record per question: question ID, the 5 votes, soft label (vote share), majority label, agreement (3/5, 4/5 or 5/5), the kept explanation, the temperature-0 label, parse failures.
4. → **Teacher check** on validation. If it fails: revise the prompt (new version name; at most 2 revisions; each revision logged), then repeat step 2.
5. After the check passes: training, test and GSM-Symbolic, with the same prompt version.
- Note: speed here is measured in bulk and does **not** count as the teacher's latency. Latency is measured separately in S11.

**S8. Student data** *(existing: `run.py build-binary-student-data`, extended)*
One training file per variant, built only from questions and teacher labels:

| Variant | Contents | For |
|---|---|---|
| `clean` | question, soft label, hard label, explanation, agreement | E1-E3, E6, E9 |
| `hard` | `clean`, trained on the hard label only | E11 |
| `noise-random-{10,20,30}` | `clean` with that share of labels flipped (fixed seed) | E4 |
| `noise-math-large`, `noise-long-large` | `clean` with every MATH (or every long) question set to large | E4 |
| `size-{250,500,1000,2000,4000}` | random subsets of `clean`, same share of each dataset (fixed seed) | E5 |
| `shuffled-explanation` | `clean` with explanations swapped between questions (fixed seed) | E7 |
| `teacher-once` | labels from the temperature-0 teacher answer | E10 |
| `oracle` | hard true labels, solvable questions only (reference baselines only, kept in a separate folder) | §9 group 5 |

Check: every non-`oracle` file is scanned for true-label fields; the build stops if one is found.

**S9. Train students** *(existing: `run.py train-student-classifier`, `train-student-dual`, `select-router-checkpoint`)*
- One config file per run (student, variant, training type, seed) in `runs/<experiment>/configs/`.
- For each run: train, pick the best checkpoint on validation (macro-F1 against the teacher), then save the "large" probability for every validation, test and GSM-Symbolic question to `predictions.jsonl`.
- A run counts as finished only when `predictions.jsonl` exists, so failed runs are re-run automatically.

**S10. Baselines** *(new: `run.py baselines`)*
- Fixed rules (always small/large, length rule, dataset rule) and random mixing: computed directly from S6, no training.
- Prompted small model (Qwen2.5-1.5B, teacher's prompt, one answer at temperature 0): run with vLLM on validation and test.
- TF-IDF and DeBERTa-v3-small on teacher labels; BERT and student on true labels (reference only): trained like S9.
- All saved in the same `predictions.jsonl` format so S12 treats every router the same way.

**S11. Speed study** *(new: `run.py latency`)*
- Nothing else running on the GPU.
- For each router (teacher temperature 0, teacher 5 votes, prompted small model, 270M, 1B, TF-IDF, DeBERTa): 20 warm-up questions, then 1,000 test questions, one at a time.
- The clock runs from receiving the question text to having a decision (tokenisation included).
- Teacher served by vLLM; students on GPU and on CPU.
- Save: p50, p95 and p99 router time; throughput at batch sizes 1, 8 and 32; router overhead as a share of the candidates' response time (TTFT and full answer, from a 100-question one-at-a-time sample); GPU-hours spent in S7-S9 for the pay-off calculation.

**S12. Evaluate** *(existing: `run.py evaluate-router`, extended)*
- Inputs: all `predictions.jsonl`, the true labels from S6, the latency results from S11.
- Computes every measure in §10, the error bars and Holm correction in §12, and the E5 line fit.
- Writes every table and figure from scripts. **No number in the paper is typed by hand.**
- Only now is the test split read.

**S13. Update the paper**
- Copy the generated tables and figures into `research/`; rewrite the text listed in §16.

### 18.4 Folder layout

```
data/
  raw/<dataset>/                         downloaded files + checksums            (S1)
  processed/<dataset>/{all,train,validation,calibration,test}.jsonl              (S2, S3)
  processed/split_manifest.json          sealed split IDs + checksums            (S3)
  oracle/<dataset>/<split>.attempts.jsonl  every candidate answer                (S5)
  oracle/<dataset>/<split>.labels.jsonl    p_small, p_large, label, unsolvable   (S6)
  teacher/<dataset>/<prompt-version>/<split>.jsonl  votes, label, explanation    (S7)
  student/<dataset>/<prompt-version>/<variant>/train.jsonl                        (S8)
runs/<experiment>/<variant>/seed_<n>/    checkpoint, predictions.jsonl, config   (S9, S10)
results/
  environment.json                                                               (S0)
  pilot/, audit/                                                                 (S4)
  latency/                                                                       (S11)
  tables/, figures/                                                              (S12)
```

Each folder written by S5-S12 contains a `manifest.json`: settings, model names,
prompt version, seed, git commit, start and end time.

### 18.5 Amount of work

| Stage | Questions | Model calls | Notes |
|---|---|---|---|
| S5 | 6,140 (train 4,000, validation 600, calibration 40, test 1,000, GSM-Symbolic 500) | 6,140 × 2 models × 5 = **61,400** | the largest GPU job; the 14B model is about 90% of it |
| S7 | 6,100 (all except calibration) | 6,100 × 6 = **36,600** | short outputs (≤150-word explanation + label) |
| S9 | – | 20 main runs + 52 extra runs | small models; hours, not days |
| S11 | 1,000 per router | 6 routers | under an hour |

S4 turns these counts into real hours for our GPU before anything large is started.

### 18.6 When something goes wrong

| Problem | What the system does |
|---|---|
| Crash during S5 or S7 | Re-run the same command; finished answers are skipped |
| An answer hits the 1,024-token limit | Kept and marked `truncated`; counted as wrong; the truncation rate is reported |
| The teacher's output cannot be read | Asked again twice, then set to "large"; the failure rate is reported |
| A check fails | The pipeline stops with a message saying which check and which fix in §11 applies |
| A prompt or setting changes | New version folder; earlier results are kept for comparison |

---

## Appendix: precise definitions

- p̂_small(x), p̂_large(x) = (number of correct answers out of 5) / 5 for each model on question x.
- A model **can solve** x if its p̂ ≥ 0.6.
- Soft true label: (p̂_small(x), p̂_large(x)).
- Hard true label: **small** if p̂_small(x) ≥ 0.6; **large** if p̂_small(x) < 0.6 ≤ p̂_large(x);
  none (unsolvable) if both are < 0.6. Unsolvable questions are left out of routing accuracy.
- End-to-end accuracy of a router = average over questions of p̂ of the model it picks.
- Cost of a question = 2 × (model parameters) × (input + output tokens), plus the router's own cost.
- PGR = (router accuracy − always-small accuracy) / (always-large accuracy − always-small accuracy).
  APGR = area under PGR plotted against the share of questions sent to large (11 points,
  trapezoid rule). Random mixing gives APGR = 0.5.
- CPT(x%) = smallest share of questions sent to large that reaches PGR ≥ x%.
- Error bars: 1,000 bootstrap resamples of test questions and seeds; 95% intervals; Holm correction within each RQ.
- Router overhead = router latency / (router latency + chosen model's response latency), reported for TTFT and for the full answer.
- Losses (paper §3): route-only = cross-entropy on teacher labels; reasoning-plus-route
  = λ_route × route loss + λ_reason × next-word loss on the teacher's explanation.
- Soft label: y_large(x) = (teacher votes for large) / 5, y_small(x) = 1 − y_large(x).
  Route loss L_route = −[y_small · log ŷ_small + y_large · log ŷ_large]. With hard labels
  y is 0 or 1 (the majority vote); with 5/5 or 0/5 agreement the two are identical.

---

## References

Only sources that were checked while writing this design are listed. "In bib" means
the entry already exists in `research/references.bib`; the others must be added
(with details copied from the source, not from memory) before they are cited in the paper.

### Methods and related work

| Work | Used for | Source | In bib |
|---|---|---|---|
| Hinton et al., 2015, *Distilling the Knowledge in a Neural Network* | Knowledge distillation; soft labels (D13) | – | ✅ `hinton2015distilling` |
| RouteLLM (Ong et al., 2024) | BERT / matrix-factorization routers; APGR and CPT measures; random-router baseline | arXiv:2406.18665 | ✅ `ong2024routellm` |
| Router-R1 (Zhang et al., 2025) | Router trained with reinforcement learning | arXiv:2506.09033 | ✅ `zhang2025routerr1` |
| Zooter (Lu et al., 2024) | Router trained on reward-model labels | – | ✅ `lu2024zooter` |
| Koehn, 2004 | Paired bootstrap | – | ✅ `koehn2004statistical` |
| RouterDC (NeurIPS 2024) | Router trained on real outcomes (mDeBERTa encoder) | arXiv:2409.19886 | ❌ add |
| Hybrid LLM (Ding et al., ICLR 2024) | Router trained on quality-score labels | arXiv:2404.14618 | ❌ add |
| AutoMix (NeurIPS 2024) | Self-verification cascade | arXiv:2310.12963 | ❌ add (related work only) |
| FrugalGPT | Cascade with a DistilBERT answer scorer | arXiv:2305.05176 | ❌ add (related work only) |
| IRT-Router (ACL 2025) | Difficulty estimation with item response theory | arXiv:2506.01048 | ❌ add (related work only) |
| Arch-Router | 1.5B router trained on LLM-generated data | arXiv:2506.16655 | ❌ add (related work only) |
| LLMRouterBench (Findings of ACL 2026) | Many routers fail to beat the best single model | arXiv:2601.07206 | ❌ add |
| Routing survey (2026) | Router categories; describes BEST-Route | arXiv:2603.04445 | ❌ add |
| BEST-Route | Router with a DeBERTa reward model | **Only seen through the survey above; find and read the original paper before citing** | ❌ |

### Models and data

| Item | Used for | Source | In bib |
|---|---|---|---|
| Qwen2.5 | Candidates and teacher; accuracies in §5.1 | Qwen2.5 release blog (qwenlm.github.io/blog/qwen2.5-llm) | ✅ `qwenteam2024qwen25` |
| Gemma 3 | Student models | – | ✅ `gemmateam2025gemma3` |
| GSM8K | Dataset | Hugging Face `openai/gsm8k` | ✅ `cobbe2021training` |
| MATH | Dataset | Hugging Face `EleutherAI/hendrycks_math` | ✅ `hendrycks2021measuring` |
| MATH-500 | MATH test set | Hugging Face `HuggingFaceH4/MATH-500` | ❌ add (cite its original source after checking the dataset card) |
| GSM-Symbolic | Extra test set (E8) | Hugging Face `apple/GSM-Symbolic`; arXiv:2410.05229 | ❌ add |
| Math-Verify | Answer checking | github.com/huggingface/Math-Verify | ❌ add (software citation) |
| vLLM on DGX Spark | Serving | build.nvidia.com/spark/vllm | – (setup only) |
| DeBERTa-v3-small, BERT-base | Baseline routers (E6, §9) | Hugging Face model cards | ❌ add |

### Added in this revision (latency measures and model-pair choice)

| Work | Used for | Source | Status |
|---|---|---|---|
| RouteLLM, model-pair choice | Strong model from the top two Chatbot Arena tiers, weak from the third; >100× price gap; also Claude 3 Opus/Sonnet and Llama 3.1 70B/8B pairs | arXiv:2406.18665 (read) | ✅ in bib |
| Hybrid LLM, model pairs | Small/medium/large quality-gap pairs (Llama-2-7B/13B, Llama-2-13B/GPT-3.5-turbo, FLAN-T5-800M/Llama-2-13B); same architecture → small gap | arXiv:2404.14618 (read) | ❌ add |
| vLLM benchmark metrics | Definitions of TTFT, TPOT, ITL, end-to-end latency | docs.vllm.ai, `benchmarks/serve` | – (method reference) |
| MLPerf Inference (Reddi et al.) | Server scenario uses the 99th-percentile latency as its limit | arXiv:1911.02549 (seen via search summary) | ❌ add after reading |
| DiSRouter | Reports routing overhead as a share of total inference time (<5%) | arXiv:2510.19208 (seen via search summary) | ❌ read before citing |
| ParetoBandit | Reports router time (p50) and overhead as a share of LLM latency | arXiv:2604.00136 (seen via search summary) | ❌ read before citing |
| Latency-aware LLM query routing | Uses TTFT as the user-perceived latency for routing | arXiv:2607.18253 (seen via search summary) | ❌ read before citing |
