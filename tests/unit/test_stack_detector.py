from cascaid.ingestion.stack_detector import detect_stack


def test_detects_langgraph_orchestrator_when_available():
    stack = detect_stack(is_available=lambda module: module == "langgraph")
    assert stack.orchestrator == "langgraph"


def test_detects_litellm_model_gateway_when_available():
    stack = detect_stack(is_available=lambda module: module == "litellm")
    assert stack.model_gateway == "litellm"


def test_detects_pinecone_vector_db_when_available():
    stack = detect_stack(is_available=lambda module: module == "pinecone")
    assert stack.vector_db == "pinecone"
