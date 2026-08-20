# Experiment Log

All route-only vs. reasoning-plus-route comparisons run in this project, in
one table. **As of 2026-08-18 the paper's primary label space is binary, not
3-way** (`research/research.md` §1/§3.2/§4.1) — experiment #4 below is now
the paper's primary result (Table 3), not an ablation; #1 is now the 3-way
generalization check (Table 4/5). #5 (binary, 1B backbone) completed
2026-08-18, closing the last incomplete cell of the label-space × backbone
grid (`runs_binary_1b/`) — see `research/research.md` §5.2/§5.3/§5.5 for how
it changed the paper's central claim (reasoning-plus-route reliably wins on
fidelity at both 1B configurations now, not just 3-way; no configuration
shows a reliable oracle-correctness advantage, including binary/1B). Rows
1–2, 4, and 5 are the paper's own reported ten-seed paired-bootstrap
results; everything else below is single-seed (42) exploratory work, not
yet run through that same multi-seed protocol. Table/section numbers in the
Source column below refer to the current `research/research.md` numbering.

Pooled test set = gsm8k + math test splits, 1,000 queries, scored against
teacher `qwen2.5-3b-v2`. Accuracy/Macro-F1 are agreement with the teacher's
decision, not ground-truth correctness (see `research/research.md` Table
4/11 for the oracle comparison instead).

| # | Experiment | Label space | Backbone | Epochs | LR schedule | Seeds | Variant | Accuracy | Macro F1 | Source |
|---|---|---|---|---:|---|---|---|---:|---:|---|
| 1 | 3-way generalization check | 3-way | 270M | 3 | linear, no warmup | 10 (0–9) | route-only | 0.549 ± 0.010 | 0.477 ± 0.025 | Table 8 |
| 1 | 3-way generalization check | 3-way | 270M | 3 | linear, no warmup | 10 (0–9) | reasoning-plus-route | 0.545 ± 0.007 | 0.460 ± 0.018 | Table 8 |
| 2 | 1B backbone (3-way) | 3-way | 1B | 3 | linear, no warmup | 10 (0–9) | route-only | 0.574 ± 0.007 | 0.505 ± 0.012 | Table 10 |
| 2 | 1B backbone (3-way) | 3-way | 1B | 3 | linear, no warmup | 10 (0–9) | reasoning-plus-route | **0.587 ± 0.014** | **0.529 ± 0.012** | Table 10 |
| 3 | Epoch-count ablation | 3-way | 270M | 20 (best=5) | linear, no warmup | 1 (42) | route-only | 0.567 | **0.519** | §5.6.4 |
| 3 | Epoch-count ablation | 3-way | 270M | 20 (best=8) | linear, no warmup | 1 (42) | reasoning-plus-route | 0.569 | 0.505 | §5.6.4 |
| 4 | Main result (binary, primary) | binary | 270M | 20 (best=2, exploratory)/3 (10-seed) | linear, no warmup | 1 (42) exploratory; 10 (0–9) confirmatory | route-only | 0.665 (n=1) / 0.652 ± 0.005 (n=10) | 0.660 (n=1) / **0.651 ± 0.005** (n=10) | Table 3 |
| 4 | Main result (binary, primary) | binary | 270M | 20 (best=6, exploratory)/3 (10-seed) | linear, no warmup | 1 (42) exploratory; 10 (0–9) confirmatory | reasoning-plus-route | **0.673** (n=1) / 0.645 ± 0.011 (n=10) | **0.672** (n=1) / 0.645 ± 0.010 (n=10) | Table 3 |
| 5 | 1B backbone (binary) | binary | 1B | 3 | linear, no warmup | 10 (0–9) | route-only | 0.652 ± 0.016 | 0.652 ± 0.016 | Table 4 |
| 5 | 1B backbone (binary) | binary | 1B | 3 | linear, no warmup | 10 (0–9) | reasoning-plus-route | **0.677 ± 0.008** | **0.676 ± 0.008** | Table 4 |

*A warmup+cosine LR-schedule check (3-way and binary, single seed) was
run and originally logged here as experiments #5/#6, but was judged not
useful enough to keep — findings didn't change any paper conclusion beyond
what the epoch-count ablation (#3) already showed, and it added a second
axis of single-seed-only variation on top of that. Removed 2026-08-18:
the whole `runs_lr_experiment/` directory deleted (~49GB freed, both
launch scripts and checkpoints), and this file isn't git-tracked, so
nothing about this experiment is recoverable except this note. To
reproduce: `train-student-classifier`/`train-student-dual` with
`--warmup-ratio 0.1 --lr-scheduler-type cosine --epochs 20 --seed 42`,
otherwise identical to `runs_epoch20`'s protocol (teacher
`qwen2.5-3b-v2`, `--dataset gsm8k math`, `--lambda-reason 0.5` for the
dual variant) — the paper's Limitations section still lists this
schedule as "not swept," which remains true.*

**Note on row 4:** unlike rows 1–2, this row now has both an exploratory
single-seed number and the confirmatory ten-seed mean, because binary's
promotion to primary label space happened after the single-seed exploratory
run but the ten-seed protocol (`multiseed/run_multiseed_binary.sh`) was
already complete by that point — see `research/research.md` §5.2/Table 3 for
the number that actually matters. The single-seed number is kept here only
because §5.6.4 discusses it directly as a cautionary example (it got the
fidelity-ranking direction backward relative to the ten-seed mean).

## Best result

**Highest raw score, across everything run:** experiment #4,
reasoning-plus-route, binary label space, linear schedule, best epoch 6 —
accuracy **0.673**, macro-F1 **0.672**. It leads on both metrics, across all
four remaining experiments, by a clear margin over the next-best binary
result (#4's own route-only row, 0.660).

**That comparison isn't apples-to-apples, though** — binary is a 2-class
problem (`cheap`/`large`) and 3-way is a 3-class problem
(`small`/`medium`/`large`); a 2-class task has a structurally higher score
ceiling (0.50 random-guess accuracy vs. 0.33), so binary rows are not
directly comparable to 3-way rows by raw score. Reading each label space on
its own terms instead:

- **Within 3-way** (now the paper's secondary/generalization label space):
  best macro-F1 is #3's route-only at epoch 5 (0.519). Route-only wins
  every 3-way single-seed comparison run so far (#1, #3) except the
  1B-backbone one (#2).
- **Within binary**: reasoning-plus-route wins on accuracy (0.673 vs
  0.665) and ties or leads on macro-F1 (0.672 vs 0.660) in #4, the only
  binary single-seed comparison logged here.
- **The statistically validated results** (ten-seed paired bootstrap, not a
  single seed) are #2 and #5: reasoning-plus-route reliably ahead of
  route-only on macro-F1 at the 1B backbone, independently at both the
  3-way (#2) and binary (#5) label spaces (95% CI excludes zero both
  times, same effect size, ≈0.024 macro-F1). Every other "win" in this
  table, including #4's raw top score, is one seed and should be read as
  directional, not confirmed.

So there are two different honest answers to "what's best," and they
mostly agree now: the single highest score overall is still a
reasoning-plus-route binary configuration (#4's single-seed exploratory
run, 0.673/0.672), but the comparisons actually validated across seeds are
also reasoning-plus-route wins, at the 1B backbone, at both label spaces
(#2, #5) — route-only's edge shows up only in the 3-way, 270M single-seed
comparisons, none of which have been confirmed past one seed. Note that
oracle-correctness tells a different story at both 1B configurations (Table
A/B above): the fidelity win for reasoning-plus-route does not carry over
into a statistically reliable oracle-correctness win at either label space.

**Bold** = the better of the two variants in that row-pair, only where the
gap is more than rounding noise.

## Oracle (ground-truth) accuracy check

Every row above is scored against the *teacher's* decision (fidelity), not
ground truth. This section re-scores every experiment's saved test
predictions against the independently constructed oracle instead (same
1,000-query pooled gsm8k+math test set, same `evaluation/oracle_check.py`
scorer Table 5/9 already use). #1 and #2 use
`multiseed/evaluate_multiseed_oracle.py` (10-seed mean ± SD, paired
bootstrap CI on the delta, same protocol as Table 5). #3 and #4 are
single-seed (42) point estimates. Sorted descending by macro-F1, highest
first. Each table also includes trivial baselines — always predicting one
fixed label regardless of the query — computed directly from the oracle's
label distribution on the same pooled test set, to check whether the
trained models are doing better than exploiting class imbalance alone
(Table 2 already flags that imbalance as substantial). 3-way baselines are
listed always-large, always-medium, always-small (macro-F1 descending,
matching the tier order small→medium→large used everywhere else); binary
baselines are always-large, always-cheap (same descending-macro-F1, same
cheap→large label-space convention).

**Table A — 3-way label space.** Latency/speedup are pooled-average
student inference time vs. the same two-call teacher (1633.2 ms, constant
across every row since it's the same teacher); #1/#2 average across the
same 10 seeds as their accuracy/macro-F1 columns.

| Rank | Experiment | Variant | Accuracy | Macro F1 | Latency | Speedup |
|---:|---|---|---:|---:|---:|---:|
| 1 | #2 (1B backbone, 10-seed) | route-only | 0.633 ± 0.010 | **0.440 ± 0.007** | 20.8 ms | 78.6x |
| 2 | #1 (main, 270M, 10-seed) | reasoning-plus-route | 0.645 ± 0.015 | 0.436 ± 0.008 | 11.3 ms | 144.3x |
| 3 | #1 (main, 270M, 10-seed) | route-only | 0.636 ± 0.016 | 0.435 ± 0.005 | 10.8 ms | 151.4x |
| 4 | #2 (1B backbone, 10-seed) | reasoning-plus-route | 0.629 ± 0.009 | 0.434 ± 0.009 | 22.4 ms | 72.8x |
| 5 | #3 (epoch ablation, 1 seed) | reasoning-plus-route | 0.610 | 0.420 | 12.1 ms | 135.0x |
| 6 | #3 (epoch ablation, 1 seed) | route-only | 0.590 | 0.420 | 11.2 ms | 145.2x |
| 7 | *baseline* | always-predict-large | 0.651 | 0.263 | — | — |
| 8 | *baseline* | always-predict-medium | 0.304 | 0.155 | — | — |
| 9 | *baseline* | always-predict-small | 0.045 | 0.029 | — | — |

**Table B — binary label space.** Rows 1–2 (#5) are 10-seed mean ± SD with
paired-bootstrap CI on the delta, same protocol as Table A's #1/#2; rows 3–4
(#4) remain single-seed (42) point estimates. Latency for #5 is the pooled
10-seed × 2-dataset average (`runs_binary_1b/`, completed 2026-08-18).

| Rank | Experiment | Variant | Accuracy | Macro F1 | Latency | Speedup |
|---:|---|---|---:|---:|---:|---:|
| 1 | #5 (1B backbone, 10-seed) | reasoning-plus-route | 0.679 ± 0.011 | **0.662 ± 0.007** | 22.3 ms | 73.1x |
| 2 | #5 (1B backbone, 10-seed) | route-only | 0.676 ± 0.014 | 0.658 ± 0.011 | 20.8 ms | 78.4x |
| 3 | #4 (linear, 1 seed) | reasoning-plus-route | 0.657 | 0.644 | 11.2 ms | 145.8x |
| 4 | #4 (linear, 1 seed) | route-only | 0.629 | 0.624 | 11.5 ms | 142.0x |
| 5 | *baseline* | always-predict-large | 0.651 | 0.394 | — | — |
| 6 | *baseline* | always-predict-cheap | 0.349 | 0.259 | — | — |

Paired delta (macro-F1, #5, 10-seed): mean $-0.004$, 95% CI $[-0.027,
+0.017]$ — not statistically reliable, unlike #5's fidelity comparison
(Table A companion above, reliable). This is the same "confirmed fidelity
win, unconfirmed correctness win" pattern #2 (1B/3-way) already showed —
see `research/research.md` §5.3 for the full discussion, including the one
way #5 differs from every other configuration: reasoning-plus-route leads
on *both* metrics' point estimates here, not just fidelity.

**Every trained variant clears its label space's best trivial baseline by a
wide margin on macro-F1** — worst trained score in Table A (#3 route-only,
0.420) still beats always-large's 0.263 by +0.157; worst trained score in
Table B (#4 route-only, 0.624) beats always-large's 0.394 by +0.230. None
of this paper's oracle-agreement results are an artifact of the models
just learning to predict the majority tier.

**The headline finding is Table A, rank 1 vs. 4.** #2 (1B backbone) is the
paper's only statistically-confirmed result — reasoning-plus-route reliably
beats route-only on teacher-agreement macro-F1 there (95% CI excludes zero,
Table 6). Against oracle ground truth, that reverses: route-only leads on
both metrics (0.440 vs. 0.434 macro-F1), and neither direction is
statistically reliable this time — the paired-delta 95% CI includes zero
for both accuracy and macro-F1, across the same 10 seeds. So the paper's
one confirmed win is a fidelity-to-the-teacher win, not a
correctness-vs-ground-truth win; it does not survive being checked against
what's actually true.

#1 (270M, 10-seed) reproduces the paper's own 3-way oracle-correctness
table almost exactly (0.636/0.645 accuracy, 0.435/0.436 macro-F1) —
expected, since the paper already reports this comparison; recomputing it
here just confirms the pipeline. Route-only wins or ties oracle ground
truth in the other 3-way comparison here (#3); reasoning-plus-route holds
up against oracle in #4's binary case (Table B, rank 1 vs. 2) — binary
remains the one setting logged here where its teacher-agreement edge
survives contact with ground truth.

## Per-tier breakdown (small / medium / large)

Only the four ten-seed comparisons (Tables 4, 5, 6, 10 in `research/research.md`)
have per-tier numbers with error bars; #3/#4 above don't break down by tier in
either the paper or this log. Values below are copied from the paper's own
per-tier prose (§5.2–§5.6), not recomputed here.

**#1 (270M) vs. teacher — F1 per tier (§5.3):**

| Tier | Route-only | Reasoning-plus-route |
|---|---:|---:|
| Small | 0.30 ± 0.07 | 0.25 ± 0.06 |
| Medium | ≈0.50 | ≈0.50 |
| Large | ≈0.63–0.64 | ≈0.63–0.64 |

**#1 (270M) vs. oracle — recall per tier, where reported (§5.4):**

| Tier | Teacher | Route-only | Reasoning-plus-route |
|---|---:|---:|---:|
| Small (recall) | 17.8% acc | 5.6% ± 2.8 pp | 4.0% ± 2.0 pp |
| Large (recall) | 49.6% | 71.9% ± 2.3 pp | 72.8% ± 2.8 pp |

Medium isn't separately reported in the paper's prose for this comparison.

**#2 (1B) vs. teacher — F1 per tier (§5.5):**

| Tier | Route-only | Reasoning-plus-route |
|---|---:|---:|
| Small | 0.33 ± 0.03 | 0.39 ± 0.03 |
| Medium | 0.53 ± 0.02 | 0.53 ± 0.04 |
| Large | 0.66 ± 0.01 | 0.67 ± 0.01 |

Reasoning-plus-route leads on every tier at this scale — opposite of #1
(270M), where route-only specifically led on small.

**#2 (1B) vs. oracle — F1 per tier (§5.6, Table 10):**

| Tier | Route-only | Reasoning-plus-route |
|---|---:|---:|
| Small | 0.078 ± 0.017 | 0.076 ± 0.019 |
| Medium | 0.500 ± 0.008 | 0.482 ± 0.023 |
| Large | 0.742 ± 0.011 | 0.745 ± 0.015 |

Route-only's oracle-correctness edge here concentrates on medium — the
opposite tier from where #1's oracle comparison found its effect (large).

**Cross-cutting pattern.** Small is the hardest tier everywhere — lowest
F1/recall in every row above, and the teacher's own small-tier F1 against
the oracle (training split, Table 3) is 0.115, the weakest number in the
whole paper. Small is also the tier where teacher-fidelity and
oracle-correctness rankings disagree most sharply for both backbones.
Large is the tier every method does best on, largely because it's the
majority label (56.4% of pooled queries, Table 2) doubling as the fallback
when nothing else succeeds (Table 2 flags 62–78% of large labels as
"nothing else worked," not "large specifically worked," §5.1) — which is
also why the always-large baseline (Tables A/B above) scores deceptively
high on raw accuracy while trailing badly on macro-F1.

## Notes per experiment

**#1/#2 (the paper's headline results).** Only these two rows have run
through the ten-seed paired-bootstrap protocol (`research/research.md`
§5.3/§5.5) that makes a comparison statistically load-bearing. #1 found no
statistically reliable difference between variants at 270M; #2 found
reasoning-plus-route reliably ahead on macro-F1 at 1B (95% CI excludes
zero). Everything below is single-seed and directional only.

**#3 (epoch-count ablation, `runs_epoch20/`).** Checked whether the paper's
3-epoch budget undertrains either variant. Both variants' validation
macro-F1 rises sharply through ~epoch 5–8 then plateaus/oscillates through
epoch 20 — the 3-epoch budget sits on the rising part of that curve. Ranking
between variants still doesn't stabilize with more training (reversed on
validation, held on test). See `research/research.md` §6.4, Table 8.

**#4 (binary label-space ablation, `runs_epoch20/*_binary/`).** Binary
routing (`{small,medium}→cheap`, `large→large`) had never been run before
this. First setting at 270M where reasoning-plus-route wins cleanly on
*both* validation and test — likely because the same "routes toward `large`
more often" behavioral shift both variants share only has two classes to
be macro-averaged over here, instead of three (see the companion analysis
artifact for the full mechanism). See `research/research.md` §6.5, Table 9.
Checked against oracle ground truth (Table B above): reasoning-plus-route
leads there too (0.657/0.644 vs. route-only's 0.629/0.624) — the one
setting logged in this file where the teacher-agreement ranking and the
oracle-correctness ranking agree in direction, unlike every ten-seed
comparison in the paper itself, where they don't.

*(A warmup+cosine LR-schedule check that used to sit here as experiments
#5/#6 was removed 2026-08-18 — see the note after the main table above.)*
