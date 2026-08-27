"""CLI: runs N executions per fault scenario, writes raw event logs + manifest.

python -m cascaid_demo.run_scenarios [--runs-per-scenario 60] [--steps 60] [--out data/runs]
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from cascaid_demo.fault_injection import SCENARIOS, make_scenario
from cascaid_demo.mock_llm_gateway import ModelGateway
from cascaid_demo.mock_vector_db import VectorStore
from cascaid_demo.pipeline import build_pipeline, export_topology
from cascaid_demo.recorder import Recorder


def run_one(run_id: str, scenario_name: str, total_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    scenario = make_scenario(scenario_name, total_steps, rng)
    recorder = Recorder()
    graph = build_pipeline()
    state = {"query": "", "retrieved_context": "", "research_notes": "", "answer": ""}
    vector_store = VectorStore()
    gateway = ModelGateway()

    for step in range(total_steps):
        config = {
            "configurable": {
                "recorder": recorder,
                "scenario": scenario,
                "step": step,
                "rng": rng,
                "vector_store": vector_store,
                "gateway": gateway,
                "run_id": run_id,
            }
        }
        state = graph.invoke(state, config=config)

    meta = {
        "run_id": run_id,
        "scenario": scenario_name,
        "total_steps": total_steps,
        "fault_onset_step": scenario.fault_onset_step,
        "cascade_step": scenario.cascade_step,
        "ramp_steps": scenario.ramp_steps,
    }
    return recorder.events, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-per-scenario", type=int, default=60)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", type=str, default="data/runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topology.json").write_text(json.dumps(export_topology(), indent=2), encoding="utf-8")

    manifest_lines = []
    total_events = 0
    seed_counter = args.seed
    for scenario_name in SCENARIOS:
        scenario_dir = out_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.runs_per_scenario):
            run_id = f"{scenario_name}-{uuid.uuid4().hex[:8]}"
            events, meta = run_one(run_id, scenario_name, args.steps, seed_counter)
            seed_counter += 1
            events_path = scenario_dir / f"{run_id}.jsonl"
            with events_path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev.to_json()) + "\n")
            manifest_lines.append(meta)
            total_events += len(events)
        print(f"{scenario_name}: {args.runs_per_scenario} runs written")

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest_lines:
            f.write(json.dumps(m) + "\n")

    print(f"Done. {len(manifest_lines)} runs, {total_events} events, written to {out_dir}")


if __name__ == "__main__":
    main()
