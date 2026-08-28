from cascaid.ingestion.runtime_context import (
    current_node,
    current_run_id,
    current_step,
    track_node,
    track_run,
    track_step,
)


def test_current_node_is_set_inside_track_node_block():
    assert current_node.get() is None
    with track_node("research_agent"):
        assert current_node.get() == "research_agent"


def test_current_node_resets_after_track_node_block():
    with track_node("research_agent"):
        pass
    assert current_node.get() is None


def test_current_run_id_is_set_inside_track_run_block():
    assert current_run_id.get() is None
    with track_run("run-1"):
        assert current_run_id.get() == "run-1"


def test_current_run_id_resets_after_track_run_block():
    with track_run("run-1"):
        pass
    assert current_run_id.get() is None


def test_current_step_is_set_inside_track_step_block():
    assert current_step.get() is None
    with track_step(3):
        assert current_step.get() == 3


def test_current_step_resets_after_track_step_block():
    with track_step(3):
        pass
    assert current_step.get() is None
