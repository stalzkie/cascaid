"""celery_adapter.py doesn't produce CallEvents itself (unlike every other adapter) --
it only keeps current_run_id/current_step/current_node correct across a Celery worker
boundary so whichever adapter runs *inside* the task body attributes correctly. Tests
drive celery's own signals directly (before_task_publish/task_prerun/task_postrun) rather
than spinning a real broker/worker -- the signals themselves are the real integration
point; only the broker/network layer is out of scope for a unit test (verified against a
real cross-process worker separately, see docs/adr/0007-celery-first-for-distributed-attribution.md).
"""

from __future__ import annotations

from celery import signals

from cascaid.ingestion.celery_adapter import instrument_celery
from cascaid.ingestion.runtime_context import (
    current_node,
    current_run_id,
    current_step,
    track_node,
    track_run,
    track_step,
)


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class _FakeTask:
    def __init__(self, headers):
        self.request = _FakeRequest(headers)


class _NoHeadersRequest:
    pass


class _NoHeadersTask:
    request = _NoHeadersRequest()


def test_before_task_publish_stashes_run_context_into_headers():
    instrument_celery()
    headers = {}
    with track_run("run-1"), track_step(2), track_node("my_agent"):
        signals.before_task_publish.send(sender="some.task", headers=headers)

    assert headers["cascaid_run_id"] == "run-1"
    assert headers["cascaid_step"] == 2
    assert headers["cascaid_node"] == "my_agent"


def test_before_task_publish_leaves_headers_untouched_when_no_context_is_set():
    instrument_celery()
    headers = {}
    signals.before_task_publish.send(sender="some.task", headers=headers)
    assert headers == {}


def test_before_task_publish_does_nothing_when_headers_is_none():
    instrument_celery()
    with track_run("run-1"), track_step(0):
        signals.before_task_publish.send(sender="some.task", headers=None)  # must not raise


def test_task_prerun_sets_context_from_headers_and_postrun_resets_it():
    instrument_celery()
    task = _FakeTask({"cascaid_run_id": "run-2", "cascaid_step": 5, "cascaid_node": "worker_agent"})

    assert current_run_id.get() is None
    signals.task_prerun.send(sender=None, task_id="t-1", task=task, args=(), kwargs={})
    assert current_run_id.get() == "run-2"
    assert current_step.get() == 5
    assert current_node.get() == "worker_agent"

    signals.task_postrun.send(sender=None, task_id="t-1", task=task, retval=None, state="SUCCESS")
    assert current_run_id.get() is None
    assert current_step.get() is None
    assert current_node.get() is None


def test_task_postrun_resets_context_even_when_the_task_failed():
    instrument_celery()
    task = _FakeTask({"cascaid_run_id": "run-3"})

    signals.task_prerun.send(sender=None, task_id="t-2", task=task, args=(), kwargs={})
    assert current_run_id.get() == "run-3"

    signals.task_postrun.send(sender=None, task_id="t-2", task=task, retval=None, state="FAILURE")
    assert current_run_id.get() is None


def test_task_prerun_does_nothing_when_headers_carry_no_cascaid_keys():
    instrument_celery()
    task = _FakeTask({})
    signals.task_prerun.send(sender=None, task_id="t-3", task=task, args=(), kwargs={})
    assert current_run_id.get() is None
    assert current_step.get() is None
    assert current_node.get() is None


def test_task_prerun_does_nothing_when_request_has_no_headers_attribute():
    instrument_celery()
    signals.task_prerun.send(sender=None, task_id="t-4", task=_NoHeadersTask(), args=(), kwargs={})
    assert current_run_id.get() is None


def test_task_postrun_is_a_no_op_for_an_unknown_task_id():
    instrument_celery()
    signals.task_postrun.send(sender=None, task_id="never-seen", task=_NoHeadersTask(), retval=None, state="SUCCESS")


def test_instrument_celery_does_not_stack_duplicate_handlers_on_repeated_calls():
    instrument_celery()
    instrument_celery()  # simulates a repeated bootstrap call

    task = _FakeTask({"cascaid_run_id": "run-5"})
    signals.task_prerun.send(sender=None, task_id="t-5", task=task, args=(), kwargs={})
    assert current_run_id.get() == "run-5"

    signals.task_postrun.send(sender=None, task_id="t-5", task=task, retval=None, state="SUCCESS")
    assert current_run_id.get() is None
