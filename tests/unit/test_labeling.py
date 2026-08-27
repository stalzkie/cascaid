from cascaid.ingestion.labeling import affected_nodes, label_step
from cascaid.ingestion.topology import build_static_graph
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES

NODE_ORDER = list(STATIC_NODES.keys())
GRAPH = build_static_graph(STATIC_NODES, ALL_EDGES)


def test_affected_nodes_rate_limit_model():
    assert affected_nodes("rate_limit_model", GRAPH) == {"primary_model", "research_agent", "synthesizer_agent"}


def test_affected_nodes_baseline_is_empty():
    assert affected_nodes("baseline", GRAPH) == set()


def test_label_step_before_onset_is_all_negative():
    labels, usable = label_step("rate_limit_model", 5, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert all(v == 0 for v in labels.values())
    assert all(usable.values())


def test_label_step_in_ramp_window_marks_affected_positive():
    labels, usable = label_step("rate_limit_model", 25, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert labels["primary_model"] == 1
    assert labels["research_agent"] == 1
    assert labels["synthesizer_agent"] == 1
    assert labels["planner_agent"] == 0
    assert all(usable.values())


def test_label_step_after_cascade_is_unusable():
    labels, usable = label_step("rate_limit_model", 31, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert all(v is False for v in usable.values())


def test_label_step_baseline_never_positive():
    labels, usable = label_step("baseline", 59, NODE_ORDER, GRAPH, fault_onset_step=None, cascade_step=None)
    assert all(v == 0 for v in labels.values())
    assert all(usable.values())
