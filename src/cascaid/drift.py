"""Model-drift CLI (PRD 7): compares recently observed feature distributions --
read back from the Graph Store's already-persisted snapshots, not a new storage
mechanism -- against the reference distribution `cascaid.train` saved alongside
the trained model, and fires an alert through the existing webhook channel when a
feature's PSI crosses the threshold.

    python -m cascaid.drift --reference models/pretrained_base.drift_reference.json
        --run-id <run_id> [--store data/graph_store] [--database-url ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cascaid.ingestion.graph_store import list_snapshots, load_snapshot
from cascaid.ingestion.schema import FEATURE_NAMES
from cascaid.serving.drift import DRIFT_THRESHOLD, compute_drift, load_reference


def check_drift(
    store_dir: str | Path, run_id: str, reference: dict, feature_names: list[str] | None = None
) -> dict[str, float]:
    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    paths = list_snapshots(store_dir, run_id)
    if not paths:
        return {}
    observed = np.concatenate([load_snapshot(p).x.numpy()[:, : len(feature_names)] for p in paths], axis=0)
    return compute_drift(reference, observed, feature_names)


def _maybe_alert(run_id: str, drifted: dict[str, float], database_url: str) -> None:
    from cascaid.alerting.dispatch import enabled_webhook_url, send_webhook
    from cascaid.alerting.rules import Alert
    from cascaid.storage.db import get_engine, make_session_factory
    from cascaid.storage.repository import init_db, record_alert

    init_db(get_engine(database_url))
    with make_session_factory(database_url)() as session:
        webhook_url = enabled_webhook_url(session)
        if not webhook_url:
            return
        for name, score in drifted.items():
            # node_type is normally one of agent/tool/model_endpoint/vector_store
            # (rules._LABEL_BY_NODE_TYPE's vocabulary) -- "drift" is deliberately
            # outside it, since this alert is about an input *feature*, not a
            # pipeline node, and never goes through rules.evaluate_risk/_message.
            alert = Alert(
                run_id=run_id,
                node_name=name,
                node_type="drift",
                risk_score=score,
                message=(
                    f"Input feature '{name}' has drifted from the training distribution "
                    f"(PSI={score:.3f}) -- consider retraining or investigating recent pipeline changes."
                ),
            )
            send_webhook(webhook_url, alert)
            record_alert(
                session, run_id=run_id, node_name=name, risk_score=score, message=alert.message, channel="webhook"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--database-url", type=str, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    reference = load_reference(args.reference)
    scores = check_drift(args.store, args.run_id, reference)

    if not scores:
        print(f"No snapshots found for run_id={args.run_id!r} under {args.store}")
        return

    print(f"Drift report for run_id={args.run_id}:")
    for name, score in scores.items():
        flag = " [DRIFTED]" if score > DRIFT_THRESHOLD else ""
        print(f"  {name}: PSI={score:.3f}{flag}")

    drifted = {name: score for name, score in scores.items() if score > DRIFT_THRESHOLD}
    if drifted and args.database_url:
        _maybe_alert(args.run_id, drifted, args.database_url)


if __name__ == "__main__":
    main()
