# Client-Readiness & "Would YC Notice This" Assessment (2026-08-30)

Note on provenance: the user referenced a prior "3 improvements" discussion
(mentioning "widening our score" as an example) that isn't recoverable from
this session, the repo's docs, memory, or GitHub issues -- I checked all
four. Rather than guess at a lost list, this is an independent senior-eng
audit of the current codebase against the PRD (`docs/Cascaid_PRD (1).md`)
and the two prior engineering logs (`GNN_Accuracy_Improvement_Log.md`,
`Auto_Instrumentation_Glue_Layer_Plan.md`), scored against this repo's own
stated decision priorities: **effectiveness, efficiency, accuracy, UX**
(`CLAUDE.md`). Every claim below was checked against the actual code, not
assumed from the PRD's aspirational language.

## 1. What's already real (don't re-litigate this)

The PRD reads like an MVP plan, but the codebase is well past MVP on most of
Section 3's core features:

- **Ingestion**: LangGraph and CrewAI are both auto-instrumented via
  `cascaid run -- <command>` (monkey-patch, zero code changes) -- not just
  hand-wired adapters. LiteLLM callbacks and Pinecone/Weaviate query patching
  are wired in automatically too. Published end-to-end: `pipx install
  cascaid` really works (`cascaid==0.2.0` is live on PyPI).
- **Model**: GATConv/GINEConv GNN, genuinely inductive (features are
  rolling per-node/edge stats + a small node-type one-hot, not node-identity
  embeddings -- verified in `snapshot_builder.py`), so it isn't
  architecturally pinned to the 7-node demo graph. 0.900±0.016 PR-AUC across
  6 fault scenarios, with a real reproducible-eval harness
  (`scripts/gnn_experiment.py`) behind that number, not a single lucky run.
- **Historical incident labeling**: Langfuse import shipped this week
  (PR #40, just merged to `staging`).
- **Dashboard**: run picker, auto-refresh, track-record view, pipeline graph
  by node type.
- **Alerting**: webhook dispatch, off-by-default / opt-in (PRD 4.6).
- **Auth**: single self-hosted admin credential, PBKDF2-hashed, session
  tokens.
- **Fast-follow items already done ahead of the PRD's own sequencing**: MCP
  server exposure (PRD §7), a Grafana JSON-datasource adapter (PRD §4.7/7),
  and model-drift monitoring (PRD §7 "watch your own model's health").
- **Packaging**: Apache-2.0 license, real PyPI metadata, single `cascaid`
  CLI replacing the old pile of `python -m` invocations.

This matters for calibrating the rest of this doc: the honest gaps left are
narrower and more specific than "what's missing from the MVP," because most
of the MVP is done.

## 2. Decision framework

Each gap below is scored effectiveness / efficiency / accuracy / UX
(`CLAUDE.md`'s own four criteria) and given one of three dispositions:
**build now**, **defer (documented)**, or **needs your call** (touches
something hard to reverse or the product's core trust promise).

## 3. The three improvements I'm prioritizing

### 3.1 Risk score calibration, not just ranking quality (build now)

**Gap**: every accuracy number in `GNN_Accuracy_Improvement_Log.md` is
PR-AUC -- a ranking metric. `risk.py` outputs `sigmoid(logits)`, which is a
valid probability-*shaped* number but nothing in the repo checks whether a
0.8 output actually corresponds to an ~80% empirical incident rate. The
PRD's Goal #1 explicitly promises "a **calibrated** probability score"
(`docs/Cascaid_PRD (1).md` §1.1) -- that's a specific, checkable claim the
project hasn't verified yet.

Why this matters more than it sounds: the whole product pitch is "trust our
number enough to page someone" (PRD §4.6, progressive trust). A model that
ranks well but is badly calibrated (e.g. everything clusters at 0.6-0.75
regardless of true risk) will pick the wrong alert threshold, undermining
exactly the trust-building the observe-only period is designed to earn.

- **Effectiveness**: high -- this is the actual thing standing between "the
  architecture works" (proven) and "the number means what we say it means"
  (unverified).
- **Efficiency**: low cost -- add a reliability-curve / Brier-score check to
  `scripts/gnn_experiment.py` (data already exists, no new pipeline), and if
  it's off, a one-line temperature-scaling or isotonic-regression fit on the
  held-out set is a small, well-understood fix.
- **Accuracy**: directly closes a PRD-stated, currently-unverified claim.
- **UX**: makes the alert-threshold picker and the dashboard's risk-band
  colors trustworthy by construction instead of by assumption.

**Decision**: build now. No new infra, no external dependency, uses data
already produced by the existing experiment harness.

### 3.2 Make Slack/PagerDuty alerts actually land correctly (build now)

**Gap**: `dispatch.py`'s own docstring says "Slack/PagerDuty are both
webhook-shaped for v1, so this one function covers all three" -- but it
POSTs Cascaid's raw `Alert` dataclass JSON. Slack's incoming-webhook format
requires a `text` (or `blocks`) key; PagerDuty's Events API v2 requires its
own `routing_key`/`event_action`/`payload` envelope. Neither will render a
readable message or open a real incident from Cascaid's current payload --
the comment describes the intent, not what the code does.

This is exactly the kind of PRD Core Feature (§3, item 5 -- not a fast-follow)
that a design partner will hit on day one and conclude "the alerting doesn't
work," which is a worse first impression than not having Slack/PagerDuty
support at all.

- **Effectiveness**: high relative to effort -- closes a literal, named
  Core Feature gap.
- **Efficiency**: very low cost -- one small payload-formatting function per
  channel (`format_slack_payload`, `format_pagerduty_payload`), reusing the
  existing `Alert` data and `send_webhook`'s transport/failure handling
  unchanged.
- **Accuracy**: N/A (not a model concern).
- **UX**: turns "an alert fired somewhere" into "a real Slack message /
  PagerDuty incident with the right node named," matching the PRD's own
  alert-copy example (§5.2).

**Decision**: build now. Config already stores a webhook URL and a channel
type is a natural one-column addition; no architecture change.

### 3.3 Scheduled retraining tied to the existing drift monitor (build now, smaller first slice)

**Gap**: PRD §4.3 promises the base model "fine-tunes quietly as real
customer data accumulates." Today that's a manual step done by hand during
a design-partner pilot (§10, Phase 3, step 13). Model-drift monitoring
(PRD §7) is already built (`drift.py`, `check_drift`/`compute_drift`), but
nothing acts on a drift signal except presumably an alert -- there's no
retrain trigger. That's fine for 1-2 hand-held pilots; it doesn't scale to
"the entire client base," where nobody on your team can manually babysit
every self-hosted install's model.

- **Effectiveness**: high for the stated goal ("usable for our entire
  client base") specifically -- this is the mechanism that lets a
  self-hosted install improve on its own instead of needing a human in the
  loop per customer.
- **Efficiency**: medium cost. The training path, data, and drift-detection
  trigger all already exist; the new work is wiring "drift-detected (or
  on a schedule) -> retrain on accumulated `IncidentLabel`/`ScoreHistory` ->
  version and hot-swap the served model" -- a real but bounded feature, not
  a research project.
- **Accuracy**: directly closes the gap between "0.90 PR-AUC on synthetic
  scenarios" and the PRD's own honest framing that real accuracy depends on
  real customer incident data (`GNN_Accuracy_Improvement_Log.md`'s "What's
  still open").
- **UX**: invisible when working, which is correct -- the whole point of
  "fine-tunes quietly."

**Decision**: build a first slice now -- a `cascaid retrain` CLI command
that reruns `train.py` against a live install's accumulated Postgres data
and atomically swaps the served model artifact on success, runnable by hand
or from cron/CI. Full "auto-fires the moment drift crosses a threshold"
wiring is a natural fast-follow once the manual command is proven, not a
prerequisite for it.

## 4. One thing that needs your call, not mine

### LLM-generated risk explanations (PRD §7) -- the highest-leverage "YC will notice this" feature, but it has a real trade-off

Right now the dashboard shows a bare number. PRD §7's own example --
"agent-checkout is at elevated risk because its vector-store dependency has
shown rising p99 retrieval latency" -- is the single most demo-able,
investor-legible feature left unbuilt: it's what turns "a threshold alarm"
into "a system that explains itself," and it's cheap to build (one prompt
constructed from data the API already computes, one LLM call, no new
storage).

**Why I'm not just building it**: Cascaid's own competitive pitch is
"self-hosted, your data never leaves your VPC" (PRD §5.2 Deployment, §8
Business Model, §11 Data sensitivity) -- and agent traces are explicitly
called out as containing raw prompts/outputs that "may include customer PII
or proprietary content" (§11). Silently wiring a call to an external LLM
API (Anthropic, OpenAI, etc.) to generate explanations would mean node
names, latency/error/cost metrics, and potentially trace fragments leave
the customer's cluster -- directly contradicting the product's stated
differentiator. That's a decision that changes what the product *is*, not
an implementation detail, so it's yours to make, not mine to default on.

Options, roughly in order of how much they preserve the VPC promise:
1. **Off by default, opt-in, customer supplies their own LLM API key/endpoint** (incl. self-hosted/local models via an OpenAI-compatible endpoint, e.g. vLLM/Ollama) -- preserves "nothing leaves your cluster unless you explicitly choose otherwise," closest to the existing trust story.
2. **On by default using a hosted LLM**, with a clear one-line disclosure in the observe-only onboarding step -- faster time-to-wow for self-serve/demo users, weaker on the VPC pitch.
3. **Skip the LLM call entirely; template-based explanations** (fill a sentence from the same neighbor-delta data, no LLM) -- zero data leaves anywhere, but noticeably less impressive than real natural-language reasoning.

My recommendation is **option 1**: it's a small config addition on top of
work you'd do anyway for option 2, keeps the differentiator intact, and
still ships the demo-able feature. Let me know if you want that, a
different option, or want to skip this one for now.

## 5. Explicitly deferred (considered, not missed)

- **Helm chart for Kubernetes** (PRD §5.2/4.4): real gap for k8s-native
  enterprise prospects, but Docker Compose already unblocks every self-serve
  and design-partner path today. Effort-to-differentiation ratio is worse
  than items 3.1-3.3; revisit once a real prospect asks for it rather than
  building speculatively.
- **RAG over incident history** (PRD §7): a natural pairing with the LLM
  explanations feature above, but adds a second piece of infra (embeddings
  + retrieval over pgvector) and inherits the same data-residency question
  at a larger scope. Sequence after 4 is resolved, not before.
- **AutoGen / other-orchestrator auto-instrumentation**: LangGraph + CrewAI
  already cover the two largest agent-orchestration frameworks
  (`Auto_Instrumentation_Glue_Layer_Plan.md` explicitly scoped AutoGen out
  of the beta pass); chasing every framework is a moving target with
  diminishing returns until a specific prospect needs it.
- **Multi-tenant SaaS control plane, SSO/RBAC/audit logs**: these are the
  PRD's own stated paid-tier, non-MVP items (`MVP_Accuracy_and_Product_Roadmap.md`
  §3) -- self-hosted-per-customer is the actual go-to-market model, so a
  single-admin-per-install auth model is correct scope, not an oversight.
- **pgvector full auto-patch**: already a deliberate, documented decision
  (`Auto_Instrumentation_Glue_Layer_Plan.md`) -- a Postgres extension has no
  distinct client library to patch safely; the one-line manual wrap is the
  right trade for now.

## 6. Sequencing

1. 3.1 (calibration check) and 3.2 (Slack/PagerDuty payloads) -- independent,
   low-risk, no external dependencies, can land in parallel.
2. 3.3 (`cascaid retrain` command) -- next, since it's the mechanism that
   lets the product scale past hand-held pilots.
3. Section 4 (LLM explanations) -- once you've picked an option, since it's
   the highest-visibility feature but the one genuine trust/architecture
   trade-off in this list.
