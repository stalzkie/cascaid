# Cascaid — Product Requirements Document

**Predictive cascading-failure intelligence for AI-native systems**

Version 0.3 — Added onboarding/time-to-value as a first-class design constraint

---

## 1. Vision & Problem Statement

Modern AI systems fail in cascades the same way microservices do — a multi-agent workflow (agent A calls agent B, which calls a tool, which queries a vector DB, which calls an LLM API) is structurally a dependency graph, and a single degraded node propagates through it exactly like a transmission-line trip propagates through a power grid. A rate-limited model API, a slow vector DB query, or one failing agent in an orchestration chain can silently degrade or collapse an entire pipeline.

Today's AI observability tools (Langfuse, LangSmith, Arize Phoenix, W&B Weave) are excellent at **tracing what happened on one call** — logging prompts, latencies, token counts, and eval scores per request. None of them model the **structural topology of the pipeline itself** to predict which part is about to take the whole system down. That's a materially different problem, and it's the same wedge Cascaid identified against Causely/Asserts in traditional microservices — except this market is newer, growing faster, and has weaker incumbents on the specific structural-prediction problem.

**Cascaid's wedge**: train a GNN on the dependency graph of an AI pipeline (agents, tools, vector stores, model endpoints) plus historical failure/degradation events, to predict cascade risk before it propagates — not just trace and explain a failure after the fact.

### 1.1 Goals
- Predict cascade risk at the node level (agent, tool call, vector DB, model endpoint) ahead of a full pipeline failure, with a calibrated probability score
- Ingest from data AI teams already generate — LangGraph/CrewAI/AutoGen execution graphs, LiteLLM/model-gateway routing logs, vector DB latency/quality metrics — zero new instrumentation required
- Ship as a self-serve, install-in-minutes product, with an on-prem/VPC option as the differentiated enterprise tier (model/agent execution data is often more sensitive than generic infra telemetry, which strengthens this pitch)
- Be honest about resolution: an MVP that beats "no signal" and a flattened baseline, not a claim of perfect foresight

### 1.2 Non-goals (for MVP)
- Full auto-remediation (automatically rerouting/retrying a failing chain) — v2+ territory, natural fit for an agent-based fast-follow (see Section 7)
- Multi-tenant/multi-pipeline federation across an org — single pipeline/team first
- Replacing existing AI observability platforms — Cascaid reads from Langfuse/LangSmith/Phoenix's existing trace data where possible, rather than requiring a new SDK
- Deep prompt/eval quality scoring — that's Langfuse/Phoenix's job; Cascaid's job is structural cascade prediction

---

## 2. Target Users & Buyer

| Persona | Role | What they want from Cascaid |
|---|---|---|
| Primary user | AI/ML Engineer running agentic or RAG pipelines in production | An early warning that a specific agent/tool/model dependency is about to take down the whole workflow, before users see it |
| Buyer | Head of AI Engineering / VP Engineering | Reduced pipeline downtime and degraded-output incidents, justifiable against the cost of silent quality regressions in customer-facing AI features |
| Secondary user | Platform engineer supporting an AI team | Wants visibility into which model/vector-DB/tool dependency is the current weak point across pipelines they support |

**ICP for first pilots**: teams running production agentic or RAG pipelines with real orchestration complexity (multi-agent chains, fallback model routing, or multi-step tool use) — not a single-call chatbot wrapper. Already instrumented with at least one of Langfuse, LangSmith, Phoenix, or raw OTel-for-LLM tracing, since that's your fastest ingestion path. Big enough to have felt a cascading degradation (a slow vector DB dragging down an entire agent chain, a rate-limited model silently degrading fallback quality) but small enough to pilot without enterprise procurement friction.

**Why this ICP over generic microservices**: this market is earlier-stage and faster-growing, the incumbents are strong at tracing but not at structural prediction, and you have a direct "I built this because I needed it" story from your own AI engineering work.

---

## 3. Core Features (MVP Scope)

1. **Pipeline topology ingestion** — reads execution graphs from LangGraph/CrewAI/AutoGen orchestration, LiteLLM/model-gateway routing logs, and vector DB (Pinecone/Weaviate/pgvector) query metrics to build a live dependency graph of the AI pipeline. Open source, self-hosted by default.
2. **Historical incident/degradation labeling** — ingests past incident timestamps and quality-degradation events (e.g. a logged eval-score drop, a spike in fallback-model usage, a retrieval-latency incident) to build training labels, sourced from Langfuse/LangSmith/Phoenix exports or manual import for MVP.
3. **Cascade risk model** — GNN trained on (pipeline topology snapshot + node/edge features: latency, error rate, token cost, retry rate, model/tool identity) → per-node risk score, refreshed on a rolling window.
4. **Risk dashboard** — visual pipeline graph (agents, tools, vector stores, model endpoints as distinct node types) colored by current risk, plus a track-record view of predicted risk vs. actual incidents.
5. **Alerting** — webhook/Slack/PagerDuty push before a threshold is crossed, framed around AI-specific failure modes (e.g. "vector DB retrieval degrading, expect downstream generation quality drop in ~10 min").
6. **Self-hosted deployment path** — Helm chart/Docker Compose so the whole pipeline runs inside the customer's own VPC — important here since agent execution traces can contain sensitive prompts/outputs, making "your data never leaves your cluster" an even stronger pitch than in generic microservices.

### Explicitly deferred to post-MVP
- Agentic auto-remediation (an agent that investigates and proposes/executes a fix — see Section 7)
- Multi-pipeline/cross-team graph stitching
- Deep prompt-quality or eval scoring (stays Langfuse/Phoenix's territory)
- Native integrations beyond LangGraph/CrewAI/LiteLLM/major vector DBs

---

## 4. Onboarding & Time-to-Value

Ease of adoption is treated as a core product requirement here, not a post-launch polish pass — LocalForge's weak launch numbers are a direct lesson that a good idea with setup friction or an unclear five-second pitch doesn't convert, regardless of technical merit. Every item below is a constraint on how Section 3's features get built, not a separate workstream.

**4.1 Zero instrumentation changes**
Cascaid reads from what a pipeline already emits — LangGraph's own execution state, LiteLLM's existing logs, vector DB client metrics — rather than requiring code changes (no `@observe` decorators or new context managers to add). Where Langfuse/LangSmith/Phoenix is already running, Cascaid reads their export instead of asking for a second SDK. This is a hard requirement on the ingestion agent design in Section 5, not just a marketing claim.

**4.2 Local demo mode — value in under five minutes, no real infra connected**
`cascaid demo` spins up the synthetic fault-injection test bed from the Phase 1 dev plan and shows a live cascade being predicted and flagged in real time, before anyone connects a real pipeline. First contact with the product costs the user nothing and requires no setup — critical for a product with no reputation yet.

**4.3 Ship pretrained, not cold**
A base model pretrained on synthetic fault-injection scenarios ships with the product and gives a usable signal from day one, fine-tuning quietly as real customer data accumulates. Directly resolves the cold-start risk already flagged in Section 10 — a self-serve product that's useless until a customer has fed it months of their own incidents doesn't survive first contact.

**4.4 One command, one artifact**
A single Docker Compose file or Helm chart stands up ingestion, model serving, storage, and dashboard together. Setup is `docker compose up` or `helm install cascaid cascaid/cascaid` — not five services the customer has to wire together themselves.

**4.5 Auto-detect the stack**
On first run, Cascaid detects whether LangGraph, CrewAI, LiteLLM, or a specific vector DB is present and configures ingestion automatically, rather than asking the user to declare their stack upfront. This turns the framework-fragmentation risk (Section 10) into an onboarding feature instead of a setup burden.

**4.6 Progressive trust: observe-only before alerting**
Default to a silent "observe and log predictions" mode for an initial period, with alerting off until the user explicitly opts in. Nobody trusts a brand-new tool's alerts on day one — this lets the track-record view build credibility before the product starts interrupting anyone, directly mitigating the alert-fatigue risk in Section 10.

**4.7 Show up where they already are**
A Grafana panel plugin and the MCP-server exposure (Section 7) both mean Cascaid's signal appears inside a screen or tool the user already has open, rather than requiring a new dashboard habit. Lower adoption bar than "check yet another tab."

---

## 5. Architecture

### 5.1 High-level system diagram (textual)

```
[Customer AI Pipeline]
   ├─ LangGraph / CrewAI / AutoGen  ──┐  (agent orchestration graph)
   ├─ LiteLLM / model gateway         │  (model routing, fallback chains)
   ├─ Vector DB (Pinecone/Weaviate/   │  (retrieval latency & quality)
   │   pgvector)                     │
   └─ Langfuse / LangSmith / Phoenix │  (existing trace export, if present)
       trace export                  │
                                      ▼
                        [Cascaid Ingestion Agent]
                        (auto-detects stack; Python/Go
                         service reading orchestration +
                         gateway + vector DB metrics into
                         graph snapshots — zero code changes
                         required on the customer's pipeline)
                                      │
                          builds rolling graph snapshots
                                      ▼
                        [Graph Store] (serialized PyG Data
                         objects, versioned by timestamp;
                         node types = agent / tool / model
                         endpoint / vector store)
                                      │
                                      ▼
                  [Model Serving API] (FastAPI + PyTorch/PyG)
                  - ships with a pretrained base model
                  - GNN inference on latest graph snapshot
                  - outputs per-node risk score + confidence
                                      │
              ┌───────────────────────┼────────────────────┐
              ▼                       ▼                     ▼
     [Postgres/Timescale]     [Alerting Service]     [Dashboard API]
     (score history,          (off by default —       (serves risk
      incident labels)         opt-in; webhook/          graph to frontend
                                Slack/PagerDuty)          + Grafana panel
                                                           + MCP server)
                                                             │
                                                             ▼
                                                  [React Dashboard]
                                                (pipeline graph viz,
                                                 track record, alerts)
```

Everything above the dashed line ships as a single Docker Compose/Helm artifact (Section 4.4) — the customer stands up one thing, not five.

### 5.2 Component detail

**Ingestion Agent**
- Auto-detects which of LangGraph/CrewAI/AutoGen, LiteLLM, and supported vector DBs are present on first run (Section 4.5), rather than requiring upfront configuration
- Reads LangGraph/CrewAI/AutoGen execution state to infer the agent/tool call graph directly from the orchestration framework's own graph definition — an advantage over generic microservices ingestion, since the topology is often already explicit in code rather than needing to be inferred purely from traces
- Reads LiteLLM/model-gateway logs for model routing, fallback events, and per-call latency/error/cost
- Reads vector DB client metrics (query latency, retrieved-result relevance scores if available)
- Where the customer already runs Langfuse/LangSmith/Phoenix, reads their existing trace export in preference to requiring a new SDK (Section 4.1)

**Graph Store**
- Serialized PyG `Data` objects indexed by timestamp, no dedicated graph database needed for MVP
- Node types explicitly typed (agent, tool, model endpoint, vector store) — different node types fail in different ways (a model endpoint rate-limits, a vector store degrades in latency, an agent loops), and this typing is itself a useful model feature

**Model Serving**
- FastAPI wrapping GATConv/GINEConv GNN — edge features include token cost and retry rate alongside latency/error rate, since cost spikes are an AI-pipeline-specific early warning signal (e.g. silent fallback to a slower/pricier model)
- Ships with a pretrained base model (Section 4.3) so day-one predictions are usable before any customer-specific fine-tuning
- Training happens offline on historical snapshots + incident labels, versioned artifact loaded by the API

**Storage**
- PostgreSQL (+ TimescaleDB) for score history, incident labels, alert history, configuration
- pgvector as the vector store for the RAG-over-incident-history feature (Section 7), reusing the same Postgres instance rather than standing up a separate vector DB

**Alerting**
- Off by default, opt-in only (Section 4.6) — simple threshold-based rule for v1, with AI-specific alert copy naming the specific model/tool/vector store at risk and the likely downstream quality impact

**Frontend**
- Pipeline graph view distinguishing node types visually, risk-colored, plus the track-record view that's central to building trust in a brand-new product
- Grafana panel plugin and MCP-server exposure as alternate surfaces (Section 4.7), so the dashboard isn't the only way in

**Deployment**
- Single Docker Compose file (local/small teams) or Helm chart (Kubernetes-native customers) bundling ingestion agent, model server, Postgres, and dashboard together (Section 4.4)
- Self-hosted/VPC tier is the enterprise differentiator — agent execution traces often contain raw prompts and outputs, which is more sensitive than generic infra telemetry and a sharper reason for a customer to insist nothing leaves their environment

---

## 6. Data & Model Details

### 6.1 Training data sources
- LangGraph/CrewAI/AutoGen execution graphs and run logs — customer already has this if using these frameworks
- LiteLLM/model-gateway routing and fallback logs — customer already has this if using a gateway
- Vector DB query metrics (Pinecone/Weaviate/pgvector) — customer already has this
- Langfuse/LangSmith/Phoenix/W&B Weave trace exports, where already in use — fastest ingestion path
- Incident/degradation labels: logged eval-score drops, fallback-usage spikes, retrieval-quality incidents — sourced from the above tools' existing exports or manual logging during a design-partner pilot
- Public supplementary data for architecture validation and the pretrained base model (Section 4.3): synthetic agent-pipeline fault-injection scenarios constructed in a local LangGraph/CrewAI demo pipeline (deliberately rate-limiting a model endpoint, degrading a vector DB), since no equivalent public AI-pipeline-cascade benchmark exists yet

### 6.2 Metrics (same underlying problem shape, carried over)
- PR-AUC as primary metric — degradation/incident events remain rare relative to normal operation
- Lead-time accuracy — how far ahead of an actual quality drop or outage the model flags elevated risk
- GNN vs. flattened baseline (XGBoost on node features without adjacency) — must beat this to justify the graph model
- Real vs. shuffled adjacency ablation
- Track live precision/recall against logged alerts once in production, feeding the track-record UI

---

## 7. AI-Engineering-Native Extensions (fast-follow, not MVP-blocking)

- **LLM-generated risk explanations**: feed the GNN's risk score, flagged node, and neighbor metric deltas into an LLM prompt to produce a plain-English explanation (e.g. "agent-checkout is at elevated risk because its vector-store dependency has shown rising p99 retrieval latency").
- **RAG over incident history**: retrieve the most similar past incident/postmortem by graph-structure + symptom similarity via pgvector, surfaced automatically when a risk alert fires.
- **Agentic remediation fast-follow**: a LangGraph/Claude Agent SDK-based agent investigates a flagged risk (checks recent deploys, model-gateway config changes, vector index freshness) and proposes — with human approval — a specific action.
- **Expose Cascaid as an MCP server**: any agent (Claude Code, an internal agent workflow, Claude Tag in Slack) can query "what's the current cascade risk on our RAG pipeline" as a direct tool call — positions Cascaid as infrastructure other AI systems consume, and doubles as an onboarding surface (Section 4.7).
- **Watch your own model's health**: a customer's topology changes as they ship new agents/tools, so the GNN's input distribution will drift — worth a lightweight model-drift check (Evidently AI or similar) before this becomes a customer-facing reliability problem for your own product.

---

## 8. Business Model
- Open-source ingestion agent/SDK — adoption driver, and especially important here since AI teams are wary of a new closed-source agent touching prompt/output data
- Usage-based hosted tier priced on pipeline nodes/traces ingested per month
- Self-hosted/VPC enterprise tier, annual license (~$20–60k/yr range) — the sensitivity of agent execution data (raw prompts, outputs, sometimes PII) makes this tier's pitch stronger than in the generic microservices version
- Land-and-expand: one pipeline, one team, then org-wide

## 9. Competitive Positioning
- **Langfuse, LangSmith, Arize Phoenix, W&B Weave**: excellent per-call tracing, prompt/eval quality tooling — but reactive, and not modeling pipeline topology for structural risk prediction
- **Causely, Grafana Asserts, Datadog Watchdog**: same "reactive/diagnostic, not predictive" gap, and largely unaware of AI-pipeline-specific node types (model endpoints, vector stores) as first-class citizens
- **Cascaid**: predictive structural cascade risk, purpose-built for agent/RAG pipeline topology, delivered with materially less setup friction than the tracing-first incumbents — a newer, faster-growing market with weaker incumbents on this specific problem

---

## 10. Step-by-Step Development Plan

### Phase 0 — Validation (1–2 weeks)
1. Talk to 5–10 AI/ML engineers running production agentic or RAG pipelines about whether "predict which part of my pipeline is about to cascade-fail" resonates — validate the pitch before building further, given the LocalForge lesson
2. Build a small LangGraph or CrewAI demo pipeline (a multi-agent RAG workflow is a good realistic test bed) and manually inject failures (rate-limit a model call, slow down the vector DB) to confirm a GNN can learn the cascade pattern — this same demo becomes the `cascaid demo` mode in Section 4.2

### Phase 1 — Core ML pipeline (3–4 weeks)
3. Build the topology-graph construction pipeline reading from your demo pipeline's LangGraph/CrewAI execution state, LiteLLM routing logs, and vector DB metrics
4. Build the baseline: GATConv/GINEConv GNN vs. flattened XGBoost, same metrics discipline (PR-AUC, adjacency ablation)
5. Validate lead-time prediction on your injected-failure scenarios; train the pretrained base model (Section 4.3) on these synthetic scenarios

### Phase 2 — MVP product shell (3–4 weeks, can overlap with Phase 1)
6. Build the ingestion agent with stack auto-detection (Section 4.5), reading LangGraph/CrewAI/AutoGen + LiteLLM + vector DB metrics, with a fallback path to Langfuse/LangSmith/Phoenix exports
7. Package everything as a single Docker Compose/Helm artifact (Section 4.4) from the start, not as a later packaging pass
8. Build the FastAPI model-serving layer shipping the pretrained base model by default
9. Build the minimal dashboard: typed pipeline graph + risk coloring + track-record log, with alerting off by default (Section 4.6)
10. Build the `cascaid demo` command as a first-class onboarding path, not an afterthought

### Phase 3 — Design partner pilot (4–6 weeks)
11. Recruit 1–2 design partners from your Phase 0 conversations — free pilot in exchange for real pipeline/incident data and a case study
12. Deploy self-hosted in their environment; this is the real test of the one-command setup and auto-detection promises in Section 4
13. Retrain/fine-tune the base model on their real pipeline data; measure against your Phase 1 metrics bar; let the customer graduate from observe-only to alerting once track record builds

### Phase 4 — Productize & launch (ongoing)
14. Tighten self-serve onboarding based on pilot friction points — treat any setup step that took a design partner more than a few minutes as a bug, not a feature gap
15. Launch the open-source ingestion agent on GitHub/Hacker News/Product Hunt/relevant AI engineering communities, leading with the one-sentence differentiation ("predicts AI pipeline cascades before they happen, not just traces them after"), the zero-instrumentation-changes claim, and the VPC/data-sovereignty story
16. Ship the MCP server exposure and Grafana panel (Sections 4.7, 7) early in this phase — both a differentiator and an onboarding surface
17. Layer in usage-based billing for the hosted tier once you have design-partner validation to point to

---

## 11. Risks & Open Questions
- **Cold-start problem**: addressed directly by shipping a pretrained base model (Section 4.3) rather than requiring months of customer-specific incident history
- **Framework fragmentation risk**: LangGraph, CrewAI, AutoGen, and raw custom orchestration all represent pipeline topology differently — MVP should pick one (LangGraph is a reasonable first choice given its explicit graph structure) as the primary target, with auto-detection (Section 4.5) making it straightforward to add others without a redesign
- **Alert fatigue risk**: mitigated by the observe-only default and opt-in alerting (Section 4.6) — a noisy risk score from day one becomes "yet another alert nobody trusts"
- **Competitive response**: Langfuse/LangSmith/Phoenix are well-funded and could add structural prediction features if this gains traction — durable advantage has to come from genuinely better prediction quality and materially lower setup friction, not just being first to use the word "predictive"
- **Data sensitivity**: agent traces often contain raw prompts/outputs, which may include customer PII or proprietary content — this raises the bar on the self-hosted/VPC story being airtight from day one, not a later hardening pass
