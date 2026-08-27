from __future__ import annotations

from cascaid.ingestion.schema import CallEvent, NodeType


class Recorder:
    def __init__(self):
        self.events: list[CallEvent] = []

    def log(
        self,
        run_id: str,
        scenario: str,
        step: int,
        caller: str,
        callee: str,
        caller_type: NodeType,
        callee_type: NodeType,
        latency_ms: float,
        error: bool,
        retried: bool,
        token_cost: float,
    ) -> None:
        self.events.append(
            CallEvent(
                run_id=run_id,
                scenario=scenario,
                step=step,
                caller=caller,
                callee=callee,
                caller_type=caller_type,
                callee_type=callee_type,
                latency_ms=latency_ms,
                error=error,
                retried=retried,
                token_cost=token_cost,
            )
        )
