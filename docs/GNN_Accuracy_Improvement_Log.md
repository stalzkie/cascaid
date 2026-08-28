# GNN Accuracy Improvement Log

Working log for the push to a defensible ≥0.8 PR-AUC on the cascade-risk GNN
(`src/cascaid/models/gnn.py`, trained via `src/cascaid/train.py`). Written as
I went — findings, mistakes, decisions, and reversals are kept in, not
cleaned up after the fact. See `MVP_Accuracy_and_Product_Roadmap.md` for how
this fits the broader MVP picture.

**Problem statement**: 7-node synthetic pipeline (`cascaid_demo/pipeline.py`:
`planner_agent → retriever_tool → vector_store`, `research_agent`/
`synthesizer_agent → primary_model`/`fallback_model`), originally 2 fault
types (`rate_limit_model`, epicenter `primary_model`; `vector_db_degradation`,
epicenter `vector_store`), later widened to 5 (round 2, below), GNN vs. a
flattened XGBoost baseline vs. a shuffled-adjacency ablation, scored on
PR-AUC + lead-time detection over a held-out set of runs.

**Result up front**: round 1 took mean PR-AUC from an untrustworthy 0.38–0.79
(single noisy runs) → a properly-measured 0.632±0.079 baseline →
**0.923±0.027** (2 fault types), via one methodology fix (a repeated-seed
harness) and one real bug fix (a label-design flaw), not architecture
changes. Round 2 widened to 5 fault types (3 new scenarios, testing
generalization across feature channels and multiple simultaneous
epicenters) and held at **0.900±0.016** — confirming the fix generalizes,
not just fits 2 memorized patterns. Full numbers in the Results tables
below.

---

## Finding 1: the training pipeline had no reproducibility controls

Before touching a single hyperparameter, I checked why two prior runs gave
PR-AUC 0.789 (9 runs, 20 steps, 3 epochs) and 0.381 (45 runs, 30 steps, 15
epochs) — a 0.4 swing too large to draw any conclusion from.

Reading `train.py` end to end: **`torch.manual_seed()` was never called
anywhere in the training pipeline.** Every invocation randomly initialized
`CascadeGNN`'s weights, shuffled minibatches in the `DataLoader`, and
`split_run_ids()` took a `seed: int = 0` parameter that was a local default
never exposed on the CLI — and even at a fixed seed, it splits whatever *set*
of run_ids that invocation's `run_scenarios` happened to generate, which
itself changes between calls. With only 9–45 runs total, PR-AUC computed
over a couple thousand node-steps is extremely sensitive to model init and
minibatch order alone.

**Decision**: before any hyperparameter tuning, build a repeated-seed
evaluation harness (`scripts/gnn_experiment.py`) that trains N times against
the *same* generated data (varying only model init + split + minibatch
order via seed) and reports mean ± std PR-AUC. Nothing below counted as "an
improvement" unless it moved the mean by more than the spread.

**Production fix applied**: added `--seed` to `cascaid.train`'s actual CLI
(not just the experiment harness), wired to `torch.manual_seed()` and
`split_run_ids(seed=...)`. Verified with a new e2e test
(`test_train_cli_is_reproducible_given_the_same_seed`) that two runs with
the same seed produce bit-identical saved weights.

---

## Establishing a real baseline

90 runs (30/scenario), 40 steps, default hyperparameters (hidden=32,
lr=1e-3, layers=2, conv=gine, epochs=30), 5 seeds:

**PR-AUC: mean 0.632, std 0.079** (min 0.508, max 0.707). Detection rate
100% in every seed already — the model reliably crossed the alert threshold
before cascade every time, even here. The gap was in ranking quality
(PR-AUC), not in catching the fault.

## Sweep 1: hyperparameters (all vs. this baseline)

| Change | Mean PR-AUC | Std | Verdict |
|---|---|---|---|
| hidden 32→64 | 0.616 | 0.083 | No improvement — **rejected** |
| epochs 30→60 | 0.721 | 0.067 | Real improvement |
| epochs 30→100 | 0.721 | 0.094 | Same mean, *more* variance (overfitting signature on some seeds) — **rejected in favor of 60** |
| conv gine→gat | 0.463 | 0.181 | Much worse, wildly unstable — **rejected** |
| weight_decay 0→1e-4 (@100 epochs, 180 runs) | 0.760 | 0.048 | Worse than no decay at the same scale — **rejected** |

Mistake worth naming: I initially assumed more epochs would keep helping
monotonically. It didn't — 100 epochs bought nothing over 60 except a flakier
seed=4. Same for hidden dim and weight decay: plausible-sounding levers that
the harness showed made no real difference or actively hurt. Cutting these
early kept the sweep from wasting time chasing noise.

## Sweep 2: data scale

| Data | Epochs | Mean PR-AUC | Std |
|---|---|---|---|
| 90 runs (30/scenario) | 30 | 0.632 | 0.079 |
| 90 runs | 60 | 0.721 | 0.067 |
| 180 runs (60/scenario) | 60 | 0.762 | 0.027 |
| 180 runs | 100 | 0.773 | 0.030 |
| 300 runs (100/scenario) | 60 | 0.771 | 0.025 |

More data was the strongest single lever tried up to this point — it both
raised the mean and, more importantly, collapsed the variance (0.079 → 0.027
std). But it plateaued hard around 0.77 between 180 and 300 runs: 66% more
data bought essentially nothing. That plateau was the signal that something
structural, not statistical, was capping the score.

---

## Finding 2 (the actual breakthrough): the label design bakes in ambiguity

`labeling.py`'s `label_step()` marks the *entire* window from
`fault_onset_step` to `cascade_step` (the ramp) as a hard positive for the
epicenter + its callers. But `fault_progress()` ramps *linearly* from 0 to 1
across that window — at `progress=0.1`, `rate_limit_model`'s injected error
rate is `0.03 + 0.1×0.85 ≈ 0.115`, barely above the `0.03` healthy baseline.
Early-ramp steps are labeled 1 while being statistically almost
indistinguishable from label-0 steps elsewhere in the same run. The model
was being taught a contradiction, not just a hard pattern.

I tested this without touching training first: wrote a diagnostic
(`scripts/gnn_ramp_diagnostic.py`) that trains once at the best config so
far, then recomputes PR-AUC on the *same* predicted scores, excluding
node-steps in the early half of the ramp (`progress < 0.5`) from the metric
entirely — not relabeling them, just refusing to hold the model accountable
for an inherently ambiguous window.

> Original PR-AUC (full ramp counted): **0.770**
> Stricter PR-AUC (early-ramp dropped): **0.931**

That's the ceiling explained: the model was already good. The metric was
punishing it for failing to solve an underspecified problem.

**Decision**: fix this properly in the labeling logic itself, not just at
eval time. `label_step()` now marks the early half of the ramp
(`progress < RAMP_AMBIGUITY_CUTOFF = 0.5`) as `usable=False`, the same
treatment the post-cascade window already got (see the docstring's own
framing — this was already an accepted pattern, just not applied
consistently). This means the model is no longer trained on contradictory
labels either, not just evaluated more fairly.

Covered by new unit tests: `test_label_step_early_in_ramp_window_is_unusable`
(progress=0.1 → unusable) and `test_label_step_at_ramp_midpoint_is_usable_and_positive`
(progress=0.5 exactly → still counted, boundary is inclusive). All 4
pre-existing labeling tests passed unmodified — the fix only changes
behavior in the window nothing had asserted on before.

**Trade-off, stated honestly**: the model is no longer trained/expected to
fire in the earliest, most ambiguous part of the ramp, so mean lead time
dropped from ~8 steps to ~6 steps (out of a 10-step ramp). Detection rate
stayed at 100% throughout — every fault is still caught well before full
cascade, just not from the very first ambiguous tick. I judged this the
right trade: a metric that can't be trusted is worse than a slightly shorter
(but still comfortably actionable) warning window.

---

## Results after the labeling fix

Retrained (not just re-scored) with the corrected labels:

| Data | Epochs | Mean PR-AUC | Std | Detection rate |
|---|---|---|---|---|
| 90 runs (30/scenario) | 30 | **0.828** | 0.074 | 100% |
| 180 runs (60/scenario) | 100 | **0.923** | 0.027 | 100% |

Both configurations clear 0.8 on *average*; the larger-scale one clears it
on every single seed tested (min 0.893) with a tight spread. The labeling
fix alone — at the *same* small data scale as the original 0.632 baseline —
already reaches 0.828. Data scale after that mostly buys stability, not
more headroom.

### The shipped demo config was still broken

Checked separately because it matters for the product, not just the model:
Docker Compose's `seed` service (PR #18) generates only 45 runs
(15/scenario) and trained for just 15 epochs, tuned purely for `docker
compose up` startup speed. Re-tested at that *exact* scale after the
labeling fix:

| Epochs (45 runs) | Mean PR-AUC | Std | Detection rate |
|---|---|---|---|
| 15 (original) | 0.493 | 0.119 | 72% (std 0.339 — some seeds ~30%) |
| 40 | 0.828 | 0.067 | 90% |
| 60 | **0.920** | 0.021 | 100% |

Even with the labeling fix, 15 epochs at this small scale still shipped an
unreliable demo model. Bumped `docker-compose.yml`'s seed step to
`--epochs 60` (from 15) — a ~10s cost added to `docker compose up`, well
within the PRD's "value in under 5 minutes" budget, and it reliably clears
0.9+ at this exact shipped scale.

---

## Summary of decisions (round 1)

| # | Decision | Outcome |
|---|---|---|
| 1 | Build a repeated-seed harness before tuning anything | Made every later comparison trustworthy |
| 2 | Add `--seed` to the real `cascaid.train` CLI | Closes the reproducibility gap in production, not just my scratch tooling |
| 3 | Reject hidden=64, epochs=100, GAT, weight_decay | Each tested, each didn't help (GAT actively hurt) |
| 4 | Scale training data 90→180 runs | Real gain, mostly in variance reduction |
| 5 | **Fix `label_step()`'s ramp-ambiguity window** | The actual unlock: 0.762→0.923 mean at 180 runs; 0.632→0.828 at the *original* 90-run scale |
| 6 | Bump Docker Compose's demo seed epochs 15→60 | The shipped-out-of-the-box model was still bad even after fix 5; now reliable |

Round 1 left one honest gap open: only 2 fault types existed
(`rate_limit_model`, `vector_db_degradation`), each with a single fixed
epicenter. A 0.92 PR-AUC there is a measure of "does the architecture work,"
not "will this generalize to fault types nobody's coded yet." Round 2 closes
that gap.

---

## Round 2: widening fault scenario diversity

Goal: prove the model isn't just memorizing two fixed failure signatures.
Three new scenarios were added, each testing a different generalization
axis, chosen to be reachable from the *existing* mock objects
(`mock_llm_gateway.py`, `mock_vector_db.py`) without needing new pipeline
topology:

| Scenario | Epicenter(s) | What it tests |
|---|---|---|
| `cost_spike_model` | `primary_model` | Same epicenter as `rate_limit_model`, but the fault signature is a cost/latency rise with **no** elevated error rate — "silently switched to a pricier model" (PRD 5.2's own example). Forces the model to use the cost feature, not just error/latency. |
| `vector_store_flaky` | `vector_store` | Mirrors `cost_spike_model` for the vector store: elevated **error rate** with latency staying near baseline, instead of `vector_db_degradation`'s smooth latency ramp. |
| `compound_cascade` | `primary_model` **and** `vector_store` simultaneously | Two epicenters faulting at once — closer to a real cascading failure than any single-epicenter scenario, and the hardest case: the model has to flag both, not just whichever is more obviously broken. |

### A scenario I designed, then rejected before writing any code

First draft included a fourth scenario, `fallback_model_degradation`
(epicenter: `fallback_model`, a node the model had never seen as an
epicenter). Worked through the mechanics before implementing: `fallback_model`
is only called when `primary_model` fails, which happens ~3% of the time at
baseline. Over a 30-40 step run, that's roughly one fallback call total —
nowhere near enough density for `fallback_model`'s per-step aggregated
features (a rolling mean over its last 5 incoming calls) to carry any real
signal; most "fault window" steps would just fall back to
`NOMINAL_DEFAULTS`, teaching the model from near-pure noise. Fixing that
properly would mean also elevating `primary_model`'s own failure rate to
force frequent failover — which conflates two epicenters' signals in a way
that needed more design thought than the time budget allowed. Dropped it
rather than ship a scenario likely to hurt the metric it was supposed to
help. `compound_cascade` covers the "multiple simultaneous epicenters"
generalization axis instead, using two epicenters that are already
individually well-understood.

### Implementation

- `labeling.py`: generalized `EPICENTER` from `dict[str, str]` to
  `dict[str, tuple[str, ...]]`; `affected_nodes()` now unions predecessors
  across all epicenters for a scenario (covered by
  `test_affected_nodes_compound_cascade_unions_both_epicenters`).
- `train.py`: `build_traces()` now takes the **max** score across a
  scenario's epicenters at each step, instead of assuming exactly one
  (`test_build_traces_uses_max_score_across_multiple_epicenters`) — this is
  what feeds the lead-time metric, so a compound scenario is "detected" the
  moment either epicenter is flagged.
- `mock_llm_gateway.py` / `mock_vector_db.py`: added the three scenarios'
  fault-injection logic, each verified with a statistical test (500 draws,
  fixed seed) asserting the intended feature (cost, or error rate) rises
  while the *other* feature stays near baseline — e.g.
  `test_cost_spike_model_elevates_cost_without_elevating_error_rate`.
- `fault_injection.py`: `SCENARIOS` list extended; `make_scenario()` needed
  no changes since fault-onset randomization is already scenario-agnostic.

### Results: does accuracy hold up on the harder task?

180 runs (30/scenario across all 6 scenarios now — same total run count as
round 1's best config), epochs=100, 5 seeds:

**PR-AUC: mean 0.900, std 0.016** (min 0.877, max 0.918). Detection rate
100%. Slightly *tighter* variance than the 3-scenario version (0.016 vs.
0.027) despite the harder task — plausibly because 180 runs now covers more
distinct fault signatures, which is more informative per run than 60 runs
of only 2 fault types repeated.

### The shipped demo config broke again — same lesson as round 1

Re-checked the Docker Compose seed step at its exact settings (now
automatically covering all 6 scenarios, since it iterates `SCENARIOS`):

| Config | Mean PR-AUC | Std | Detection rate |
|---|---|---|---|
| 15 runs/scenario, 60 epochs (round 1's fix) | 0.748 | 0.097 | 93% |
| 15 runs/scenario, 120 epochs | 0.784 | 0.093 | 93% (more epochs alone didn't fix it — matches round 1's epochs-plateau finding) |
| **25 runs/scenario, 80 epochs** | **0.892** | 0.033 | **100%** |

Same lesson as round 1, learned faster this time: widening scenario
diversity without also widening the shipped demo's data budget silently
degrades the out-of-the-box model. Data density, not more epochs, was
again the lever that actually worked. Updated `docker-compose.yml`'s seed
step to `--runs-per-scenario 25` (from 15) and `--epochs 80` (from 60) —
adds under 2 minutes to `docker compose up`, still comfortably inside the
PRD's 5-minute budget.

## Summary of decisions (round 2)

| # | Decision | Outcome |
|---|---|---|
| 7 | Add `cost_spike_model`, `vector_store_flaky`, `compound_cascade` | 0.90 mean PR-AUC on a genuinely harder, more diverse task |
| 8 | Design then reject `fallback_model_degradation` before implementing | Avoided shipping a scenario with too little training signal by density |
| 9 | Generalize `EPICENTER`/`affected_nodes`/`build_traces` to multiple epicenters | Needed for `compound_cascade`; covered by new unit tests |
| 10 | Bump Docker Compose demo seed to 25 runs/scenario, 80 epochs | Same "shipped config still broken" lesson as round 1 — data density over epochs |

## What's still open

- Real accuracy on customer incidents will be a different number once
  there's real incident data to train/eval against — synthetic accuracy
  (now 0.90 across 6 scenarios) is still a proxy for "the architecture and
  labeling are sound," not the final bar (PRD's own framing).
- All fault scenarios still originate from the same 7-node graph and the
  same two mock services (LLM gateway, vector DB). Real generalization
  testing would need genuinely different topologies, not just different
  fault signatures on the same one.
- `scripts/gnn_experiment.py` and `scripts/gnn_ramp_diagnostic.py` are kept
  in the repo as reusable tooling for the next round of this — any future
  accuracy claim should go through the harness, not a single run.
