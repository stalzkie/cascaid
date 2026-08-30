from cascaid.ingestion.stack_detector import detect_stack


def test_detects_langgraph_orchestrator_when_available():
    stack = detect_stack(is_available=lambda module: module == "langgraph")
    assert stack.orchestrators == frozenset({"langgraph"})


def test_detects_crewai_orchestrator_when_available():
    stack = detect_stack(is_available=lambda module: module == "crewai")
    assert stack.orchestrators == frozenset({"crewai"})


def test_detects_both_orchestrators_independently_when_both_are_available():
    # cascaid itself always has langgraph installed (its own demo pipeline needs
    # it), so a customer whose actual app uses only CrewAI would still have
    # langgraph importable in that same environment -- detection can't be
    # exclusive, or CrewAI would never be reachable in any real install.
    stack = detect_stack(is_available=lambda module: module in ("langgraph", "crewai"))
    assert stack.orchestrators == frozenset({"langgraph", "crewai"})


def test_detects_no_orchestrator_when_neither_is_available():
    stack = detect_stack(is_available=lambda module: False)
    assert stack.orchestrators == frozenset()


def test_detects_litellm_model_gateway_when_available():
    stack = detect_stack(is_available=lambda module: module == "litellm")
    assert stack.model_gateway == "litellm"


def test_detects_pinecone_vector_db_when_available():
    stack = detect_stack(is_available=lambda module: module == "pinecone")
    assert stack.vector_db == "pinecone"
