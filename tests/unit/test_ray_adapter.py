"""ray_adapter.py doesn't produce CallEvents itself (same shape as celery_adapter.py) --
it only keeps current_run_id/current_step/current_node correct across a real Ray worker
process boundary so whichever adapter runs *inside* the task attributes correctly. Tests
drive a real local Ray cluster (ray.init()/.remote()/ray.get()) rather than mocking
Ray's internals -- see docs/adr/0008-ray-distributed-attribution.md for why: the whole
design here exists because of two non-obvious cross-process serialization footguns that
only a real worker boundary surfaces (a mock would hide exactly the failure modes this
adapter has to avoid).
"""

from __future__ import annotations

import ray

from cascaid.ingestion import runtime_context as rc
from cascaid.ingestion.ray_adapter import instrument_ray
from cascaid.ingestion.runtime_context import track_node, track_run, track_step


@ray.remote
def _plain_add(a, b):
    # A task representative of real customer code: no reference to any cascaid
    # internals at all. Proves wrapping doesn't break unrelated tasks' pickling.
    return a + b


@ray.remote
def _report_context():
    # Reads current_run_id/current_step/current_node from a code path unconnected to
    # its own arguments -- simulates an adapter (e.g. litellm_adapter.py) running
    # inside the task body. Module-qualified access (rc.current_run_id), not a bare
    # `from runtime_context import current_run_id` -- see ray_adapter.py's module
    # docstring: a bare ContextVar reference inside anything Ray has to pickle breaks
    # serialization for the whole task, module/function references don't.
    return (rc.current_run_id.get(), rc.current_step.get(), rc.current_node.get())


def setup_module():
    instrument_ray()
    ray.init(num_cpus=2, include_dashboard=False)


def teardown_module():
    ray.shutdown()


def test_plain_task_with_no_cascaid_reference_still_pickles_and_runs():
    with track_run("run-1"), track_step(0):
        result = ray.get(_plain_add.remote(1, 2))
    assert result == 3


def test_context_propagates_to_a_real_worker_process():
    with track_run("run-2"), track_step(3), track_node("worker_agent"):
        result = ray.get(_report_context.remote())
    assert result == ("run-2", 3, "worker_agent")


def test_context_is_none_in_the_worker_when_nothing_was_set_on_the_driver():
    result = ray.get(_report_context.remote())
    assert result == (None, None, None)


def test_context_does_not_leak_into_a_call_made_without_active_tracking():
    with track_run("run-3"), track_step(1):
        ray.get(_report_context.remote())

    result = ray.get(_report_context.remote())
    assert result == (None, None, None)


def test_instrument_ray_does_not_double_wrap_on_repeated_calls():
    instrument_ray()  # simulates a repeated bootstrap call
    with track_run("run-4"), track_step(2), track_node("agent_x"):
        result = ray.get(_report_context.remote())
    assert result == ("run-4", 2, "agent_x")
