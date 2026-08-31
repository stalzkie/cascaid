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


def test_detects_anthropic_direct_sdk_when_available():
    stack = detect_stack(is_available=lambda module: module == "anthropic")
    assert stack.direct_sdks == frozenset({"anthropic"})


def test_detects_anthropic_direct_sdk_independently_of_litellm_model_gateway():
    # A pipeline can use litellm for most calls and a raw anthropic.Client()
    # elsewhere -- these are orthogonal facts, not competing answers to one
    # question (see docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md),
    # so both must be detected together, unlike model_gateway/vector_db's
    # exclusive next().
    stack = detect_stack(is_available=lambda module: module in ("litellm", "anthropic"))
    assert stack.model_gateway == "litellm"
    assert stack.direct_sdks == frozenset({"anthropic"})


def test_detects_no_direct_sdk_when_none_available():
    stack = detect_stack(is_available=lambda module: False)
    assert stack.direct_sdks == frozenset()


def test_detects_openai_direct_sdk_when_available():
    stack = detect_stack(is_available=lambda module: module == "openai")
    assert stack.direct_sdks == frozenset({"openai"})


def test_detects_anthropic_and_openai_direct_sdks_independently():
    stack = detect_stack(is_available=lambda module: module in ("anthropic", "openai"))
    assert stack.direct_sdks == frozenset({"anthropic", "openai"})


def test_detects_gemini_direct_sdk_when_available():
    # google-genai's importable package is google.genai, not literally "gemini" --
    # is_available is probed with the real module path, direct_sdks still reports
    # the friendly vendor label.
    stack = detect_stack(is_available=lambda module: module == "google.genai")
    assert stack.direct_sdks == frozenset({"gemini"})
