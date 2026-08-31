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
one question. The OpenAI adapter shipped later into that same `direct_sdks` field, once
its dedup mechanism existed. This lets a pipeline that mixes litellm for most calls with
a raw `openai.Client()` elsewhere get both observed without duplicates -- the "framework
fragmentation" case PRD 4.5 exists to catch -- rather than silently excluding the OpenAI
adapter whenever litellm is present at all.

**Update: OpenAI adapter shipped (`openai_adapter.py`).** The dedup mechanism ended up
being a contextvar reentrancy flag, not a metadata marker as originally guessed above.
Empirically verified (a real `litellm.completion()` call driven through a patched
`Completions.create`, see `test_inside_litellm_dispatch_is_true_only_around_the_real_dispatch`
in `test_litellm_adapter.py`): litellm's OpenAI provider path calls
`openai_client.chat.completions.with_raw_response.create(...)`, never `.create()`
directly, but `with_raw_response` is a `@cached_property` that looks up
`completions.create` dynamically at first access -- so it resolves through our patch of
`Completions.create` too, and a metadata marker would have needed to survive litellm's
internal `data` dict construction, which isn't guaranteed. Instead,
`litellm_adapter.inside_litellm_dispatch` (a `ContextVar[bool]`) is set `True` only
around the actual dispatch inside `patched_completion`/`patched_acompletion`, and
`openai_adapter`'s patched `create()` checks it and calls straight through (no sink)
when set -- litellm's own callback registry is the one that records that call. A direct
`openai.OpenAI().chat.completions.create()` call elsewhere is unaffected, since the flag
is only true during litellm's own dispatch window.

Also confirmed empirically this round: litellm's Gemini/Vertex path (`llms/gemini/`,
`llms/vertex_ai/`) has no `google.generativeai`/`google.genai` import anywhere -- same
situation as Anthropic, no double-counting risk, so `gemini_adapter.py` shipped as a
plain adapter alongside these two, no dedup mechanism needed.

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
- `litellm_adapter.py` now owns a small piece of state (`inside_litellm_dispatch`) that
  exists purely for `openai_adapter.py` to read -- a real, if narrow, coupling between
  two adapters that were otherwise fully independent. Dedup coverage assumes litellm's
  real dispatch happens on the same thread/task that set the flag; a background
  ThreadPoolExecutor dispatch (litellm's documented behavior for streaming sync calls,
  see `litellm_adapter.py`'s module docstring) wouldn't propagate it. Streaming isn't
  wired up on either adapter yet, so this doesn't bite today, but would need addressing
  before streaming support is added to either side of this pair.
