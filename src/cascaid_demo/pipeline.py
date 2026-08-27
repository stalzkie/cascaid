"""Small LangGraph multi-agent RAG demo pipeline (PRD Phase 0 step 2).

The LangGraph control-flow edges (planner -> retriever -> research ->
synthesizer) give the agent/tool topology "for free" from the graph
definition itself (PRD 5.2's ingestion advantage over generic
microservices). Service calls to the vector store and model gateway are not
LangGraph nodes -- they're external dependencies invoked from within a node,
recorded separately the way a real ingestion agent would read LiteLLM logs
and vector DB client metrics.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from cascaid.ingestion.schema import NodeType

STATIC_NODES: dict[str, NodeType] = {
    "planner_agent": NodeType.AGENT,
    "retriever_tool": NodeType.TOOL,
    "vector_store": NodeType.VECTOR_STORE,
    "research_agent": NodeType.AGENT,
    "primary_model": NodeType.MODEL_ENDPOINT,
    "fallback_model": NodeType.MODEL_ENDPOINT,
    "synthesizer_agent": NodeType.AGENT,
}

CONTROL_EDGES: list[tuple[str, str]] = [
    ("planner_agent", "retriever_tool"),
    ("retriever_tool", "research_agent"),
    ("research_agent", "synthesizer_agent"),
]

SERVICE_EDGES: list[tuple[str, str]] = [
    ("retriever_tool", "vector_store"),
    ("research_agent", "primary_model"),
    ("research_agent", "fallback_model"),
    ("synthesizer_agent", "primary_model"),
    ("synthesizer_agent", "fallback_model"),
]

ALL_EDGES: list[tuple[str, str]] = CONTROL_EDGES + SERVICE_EDGES


class PipelineState(TypedDict):
    query: str
    retrieved_context: str
    research_notes: str
    answer: str


def _planner_node(state: PipelineState, config: RunnableConfig) -> dict:
    return {}


def _retriever_node(state: PipelineState, config: RunnableConfig) -> dict:
    c = config["configurable"]
    ev = c["vector_store"].query(c["step"], c["scenario"], c["rng"])
    c["recorder"].log(
        run_id=c["run_id"],
        scenario=c["scenario"].name,
        step=c["step"],
        caller="retriever_tool",
        callee="vector_store",
        caller_type=NodeType.TOOL,
        callee_type=NodeType.VECTOR_STORE,
        **ev,
    )
    return {"retrieved_context": f"ctx@step{c['step']}"}


def _research_node(state: PipelineState, config: RunnableConfig) -> dict:
    c = config["configurable"]
    events, used = c["gateway"].call(c["step"], c["scenario"], c["rng"])
    for callee, ev in events:
        c["recorder"].log(
            run_id=c["run_id"],
            scenario=c["scenario"].name,
            step=c["step"],
            caller="research_agent",
            callee=callee,
            caller_type=NodeType.AGENT,
            callee_type=NodeType.MODEL_ENDPOINT,
            **ev,
        )
    return {"research_notes": f"notes(used={used})"}


def _synthesizer_node(state: PipelineState, config: RunnableConfig) -> dict:
    c = config["configurable"]
    events, used = c["gateway"].call(c["step"], c["scenario"], c["rng"])
    for callee, ev in events:
        c["recorder"].log(
            run_id=c["run_id"],
            scenario=c["scenario"].name,
            step=c["step"],
            caller="synthesizer_agent",
            callee=callee,
            caller_type=NodeType.AGENT,
            callee_type=NodeType.MODEL_ENDPOINT,
            **ev,
        )
    return {"answer": f"answer(used={used})"}


def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("planner", _planner_node)
    g.add_node("retriever", _retriever_node)
    g.add_node("research", _research_node)
    g.add_node("synthesizer", _synthesizer_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "research")
    g.add_edge("research", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def export_topology() -> dict:
    return {
        "nodes": {name: t.value for name, t in STATIC_NODES.items()},
        "edges": [[c, cal] for c, cal in ALL_EDGES],
    }
