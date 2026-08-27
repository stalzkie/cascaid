from cascaid.ingestion.runtime_context import current_node, track_node


def test_current_node_is_set_inside_track_node_block():
    assert current_node.get() is None
    with track_node("research_agent"):
        assert current_node.get() == "research_agent"


def test_current_node_resets_after_track_node_block():
    with track_node("research_agent"):
        pass
    assert current_node.get() is None
