from cascaid.ingestion.topology import build_static_graph
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES


def test_static_graph_has_expected_nodes_and_edges():
    g = build_static_graph(STATIC_NODES, ALL_EDGES)
    assert set(g.nodes) == set(STATIC_NODES.keys())
    assert g.number_of_edges() == len(ALL_EDGES)


def test_predecessors_of_primary_model():
    g = build_static_graph(STATIC_NODES, ALL_EDGES)
    preds = set(g.predecessors("primary_model"))
    assert preds == {"research_agent", "synthesizer_agent"}


def test_predecessors_of_vector_store():
    g = build_static_graph(STATIC_NODES, ALL_EDGES)
    preds = set(g.predecessors("vector_store"))
    assert preds == {"retriever_tool"}
