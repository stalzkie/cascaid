# MVP Accuracy & Product Roadmap Discussion (2026-08-28)

Summary of a planning discussion after the backend/frontend/deployment MVP work
(#12–#19) landed on `master`. Three questions: what's left for a defensible
"0.8 accuracy" MVP claim, what's the UI plan, and how does the open-source
core differ from the paid product.

## 1. Accuracy: where things actually stand

Observed PR-AUC numbers going into this discussion were noisy and inconsistent,
**not** a stable ≥0.8:

| Run | Data | Epochs | GNN PR-AUC | Baseline (XGBoost) PR-AUC |
|---|---|---|---|---|
| e2e test (`test_serve_cli.py`) | 9 runs (3/scenario), 20 steps | 3 | 0.789 | — |
| Docker seed (CI, PR #18) | 45 runs (15/scenario), 30 steps | 15 | **0.381** | 0.365 |

The second run is the more representative sample size and is a red flag: the
GNN barely beat the flattened baseline, and the shuffled-adjacency ablation
had a *higher* detection rate than real adjacency in that run — meaning graph
structure wasn't reliably contributing signal. Two PR-AUC values 0.38 apart
between runs of the same codebase also means the *evaluation itself* is not
trustworthy yet — before touching hyperparameters, the measurement needs to
stop being noisy.

Planned path to a defensible 0.8, in order:
1. **Fix evaluation noise first.** Small validation sets (a handful of runs)
   make PR-AUC swing wildly run to run. Need repeated splits / multiple
   seeds, averaged with a spread, before trusting any single number.
2. **Widen synthetic fault coverage.** Only two fault types exist today
   (`rate_limit_model`, `vector_db_degradation`), each with a single fixed
   epicenter in a 7-node graph. More fault diversity is needed for the
   accuracy claim to mean anything beyond "memorized two failure modes."
3. **Real hyperparameter sweep** — hidden dim, layers, GAT vs. GINE, epochs,
   learning rate are all still untouched defaults.
4. **Acknowledge the ceiling**: real production accuracy will be decided by
   real customer incident data once there's a design partner. Synthetic
   fault-injection accuracy is a proxy for "the architecture works," not the
   final bar.

Full experimentation log for this work: see `GNN_Accuracy_Improvement_Log.md`
in this vault.

## 2. UI plan

What exists (PR #17): a minimal pipeline risk graph + track-record chart,
single-run text-input, no auth, no live refresh. Gaps against the PRD's own
stated asks (Section 4.6/4.7):

- No run picker — there's no `GET /runs` endpoint, only "type a run_id you
  already know."
- No live refresh (poll or push) — the dashboard requires a manual click.
- No alerting-config UI — `cascaid.alerting.configure` is CLI-only today.
- No Grafana panel plugin or MCP server exposure — both explicitly named in
  PRD 4.7 as alternate surfaces ("show up where they already are"), neither
  built.
- No auth — fine for a local demo, not fine the moment this leaves localhost.

Sequencing decided: run picker + live refresh first (cheapest, closes the
biggest "feels unfinished" gap) → auth (blocking for anything beyond local
demo) → Grafana/MCP surfaces (adoption-widening; a design partner will likely
ask for at least one).

## 3. Open-source core vs. paid product

The PRD's own Non-Goals section (§1.2) already drew this line — the natural
paid-tier features are exactly what it deferred for MVP.

**Stays open** (self-hosted core, what's built today): ingestion agent, Graph
Store, single-pipeline Model Serving + pretrained base model, basic webhook
alerting, single-pipeline dashboard, Docker Compose deploy. This is the
adoption engine — self-hosted-by-default is the PRD's stated wedge against
Datadog-style SaaS, and Grafana/MCP surfaces should stay open too since
they're distribution surfaces, not monetizable features on their own.

**Natural paid tier**:
- Multi-pipeline / multi-tenant federation across an org (explicitly a PRD
  non-goal for MVP — that's the signal it's the paid tier's core value)
- Continuous fine-tuning / retraining on a customer's own incident history at
  scale, model-drift monitoring (PRD §7)
- LLM-generated risk explanations + RAG-over-incident-history (PRD §7)
- Agentic remediation (PRD §7)
- SSO / RBAC / audit logs — enterprise table stakes, none of which exist
  today
- Managed/hosted control plane, as an alternative to self-hosting
- Support SLA, compliance posture (SOC2, etc.)

This is a standard open-core split (GitLab CE/EE, Grafana OSS/Enterprise
shape), grounded in decisions the PRD had already implicitly made rather than
invented fresh.

## Immediate next step

Decided: focus next on ML accuracy specifically (GNN), since it's the actual
long pole for the "0.8 accuracy MVP" claim — not more product surface area.
