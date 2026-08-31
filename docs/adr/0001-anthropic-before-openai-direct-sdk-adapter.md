---
status: accepted
---

# Build the direct Anthropic SDK adapter before the OpenAI one, and design the OpenAI adapter to coexist with litellm rather than exclude it

litellm's OpenAI provider path internally instantiates real `openai.OpenAI`/`AsyncOpenAI`
clients and calls their `.chat.completions.create()` (confirmed in
`litellm/llms/openai/openai.py`). A global monkeypatch of the OpenAI SDK -- the only
viable interception point, since unlike litellm it has no callback registry -- would
therefore also fire for calls litellm makes on the customer's behalf, double-counting
every OpenAI-model call routed through litellm once both adapters are active. litellm's
Anthropic path has no equivalent (no `anthropic` SDK import anywhere under
`litellm/llms/anthropic/`), so a direct Anthropic adapter has no such hazard and can ship
immediately as a same-shape sibling to `litellm_adapter.py`.

We're building the Anthropic adapter first for exactly that reason -- not because
Anthropic usage is more common, but because it's the one with no unresolved correctness
question blocking it. Detection for it lands as a new `DetectedStack.direct_sdks:
frozenset[str]` field, independent of `model_gateway` (which stays specifically "which
gateway/proxy is present" -- today only litellm) -- mirroring how `orchestrators` is
already detected independently rather than exclusively, since "litellm present" and
"anthropic SDK present" are orthogonal facts about a pipeline, not competing answers to
one question. The OpenAI adapter ships later into that same `direct_sdks` field, once its
dedup mechanism exists: its patched `create()` will check for cascaid's own litellm
metadata marker on a call and skip it if litellm's own adapter already sinked a
`CallEvent` for it. This lets a pipeline that mixes litellm for most calls with a raw
`openai.Client()` elsewhere get both observed without duplicates -- the "framework
fragmentation" case PRD 4.5 exists to catch -- rather than silently excluding the OpenAI
adapter whenever litellm is present at all.

## Considered Options

- **Build OpenAI first** (rejected): larger real-world usage, but ships with the
  double-counting bug from day one; the dedup design has to exist before the adapter is
  correct, so building it first doesn't actually save time.
- **Mutual exclusion** (rejected for OpenAI): only activate the OpenAI direct adapter
  when litellm is not detected at all. Simpler, but silently blind to any raw
  `openai.Client()` call in a pipeline that also happens to use litellm anywhere --
  common in practice, not a hypothetical edge case.

## Consequences

- `DetectedStack` gains a `direct_sdks: frozenset[str]` field alongside `orchestrators`
  and `model_gateway`; `_instrument_bootstrap.py` wires each direct-SDK adapter off of it
  independently of whichever `model_gateway` is also detected.
