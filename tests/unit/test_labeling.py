from datetime import datetime, timedelta, timezone

from cascaid.ingestion.labeling import affected_nodes, label_step, label_step_from_incidents
from cascaid.ingestion.topology import build_static_graph
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES

NODE_ORDER = list(STATIC_NODES.keys())
GRAPH = build_static_graph(STATIC_NODES, ALL_EDGES)


def test_affected_nodes_rate_limit_model():
    assert affected_nodes("rate_limit_model", GRAPH) == {"primary_model", "research_agent", "synthesizer_agent"}


def test_affected_nodes_baseline_is_empty():
    assert affected_nodes("baseline", GRAPH) == set()


def test_affected_nodes_compound_cascade_unions_both_epicenters():
    # compound_cascade has two simultaneous epicenters (primary_model,
    # vector_store) -- affected_nodes must union both epicenters' own node
    # and predecessors, not just the first.
    assert affected_nodes("compound_cascade", GRAPH) == {
        "primary_model",
        "research_agent",
        "synthesizer_agent",
        "vector_store",
        "retriever_tool",
    }


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


def test_label_step_early_in_ramp_window_is_unusable():
    # progress = (21-20)/10 = 0.1 -- fault_progress is near zero here, so
    # this step is statistically indistinguishable from healthy even though
    # it falls inside [fault_onset_step, cascade_step). Excluded rather than
    # mislabeled, the same way the post-cascade window already is.
    labels, usable = label_step("rate_limit_model", 21, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert all(v is False for v in usable.values())


def test_label_step_at_ramp_midpoint_is_usable_and_positive():
    # progress = (25-20)/10 = 0.5 -- exactly at the cutoff, still counted.
    labels, usable = label_step("rate_limit_model", 25, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert labels["primary_model"] == 1
    assert all(usable.values())


def test_label_step_after_cascade_is_unusable():
    labels, usable = label_step("rate_limit_model", 31, NODE_ORDER, GRAPH, fault_onset_step=20, cascade_step=30)
    assert all(v is False for v in usable.values())


def test_label_step_baseline_never_positive():
    labels, usable = label_step("baseline", 59, NODE_ORDER, GRAPH, fault_onset_step=None, cascade_step=None)
    assert all(v == 0 for v in labels.values())
    assert all(usable.values())


T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)


def test_label_step_from_incidents_marks_node_positive_within_window():
    incidents = [("primary_model", T0)]
    labels, usable = label_step_from_incidents(NODE_ORDER, incidents, step_start=T0, step_end=T0)

    assert labels["primary_model"] == 1
    assert labels["vector_store"] == 0
    assert all(usable.values())


def test_label_step_from_incidents_ignores_incidents_outside_window():
    incident_far_away = T0 + timedelta(hours=1)
    labels, _ = label_step_from_incidents(NODE_ORDER, [("primary_model", incident_far_away)], step_start=T0, step_end=T0)

    assert labels["primary_model"] == 0


def test_label_step_from_incidents_respects_window_before_and_after():
    just_inside_before = T0 - timedelta(minutes=5)
    labels, _ = label_step_from_incidents(
        NODE_ORDER,
        [("primary_model", just_inside_before)],
        step_start=T0,
        step_end=T0,
        window_before=timedelta(minutes=5),
        window_after=timedelta(minutes=5),
    )

    assert labels["primary_model"] == 1


def test_label_step_from_incidents_is_node_local_not_propagated_to_predecessors():
    # No affected_nodes()-style propagation to callers -- see
    # docs/Real_Data_Retraining_Plan.md: propagating a real incident to
    # predecessors would bake in an unverified assumption from the synthetic
    # scenarios' calibration, not something known about a real incident.
    labels, _ = label_step_from_incidents(NODE_ORDER, [("primary_model", T0)], step_start=T0, step_end=T0)

    assert labels["research_agent"] == 0
    assert labels["synthesizer_agent"] == 0


def test_label_step_from_incidents_unusable_without_wall_clock_bounds():
    labels, usable = label_step_from_incidents(NODE_ORDER, [("primary_model", T0)], step_start=None, step_end=None)

    assert all(v is False for v in usable.values())
    assert all(v == 0 for v in labels.values())


def test_label_step_from_incidents_ignores_unknown_node_names():
    labels, usable = label_step_from_incidents(NODE_ORDER, [("not_a_real_node", T0)], step_start=T0, step_end=T0)

    assert all(v == 0 for v in labels.values())
    assert all(usable.values())
