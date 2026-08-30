"""Unit seam: CallEvent's JSON round trip, including the optional occurred_at
wall-clock timestamp (needed to map a real IncidentLabel onto a snapshot --
see docs/Real_Data_Retraining_Plan.md). Additive/optional so JSONL logs
captured before this field existed still parse."""

from datetime import datetime, timezone

from cascaid.ingestion.schema import CallEvent, NodeType


def _event(**overrides) -> CallEvent:
    defaults = dict(
        run_id="run-1",
        scenario="rate_limit_model",
        step=0,
        caller="agent",
        callee="primary_model",
        caller_type=NodeType.AGENT,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=100.0,
        error=False,
        retried=False,
        token_cost=0.01,
        occurred_at=None,
    )
    defaults.update(overrides)
    return CallEvent(**defaults)


def test_round_trips_occurred_at_through_json():
    ts = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    event = _event(occurred_at=ts)

    restored = CallEvent.from_json(event.to_json())

    assert restored.occurred_at == ts


def test_from_json_defaults_occurred_at_to_none_for_old_logs_without_the_field():
    event = _event()
    payload = event.to_json()
    del payload["occurred_at"]

    restored = CallEvent.from_json(payload)

    assert restored.occurred_at is None
