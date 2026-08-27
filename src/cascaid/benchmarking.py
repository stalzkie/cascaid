"""Benchmark result visualization + historical archiving for cascaid.train runs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_comparison_chart(results: dict[str, tuple[float, dict]], dest_path: Path) -> None:
    names = list(results.keys())
    pr_aucs = [results[n][0] for n in names]
    detection_rates = [results[n][1]["detection_rate"] for n in names]
    lead_times = [results[n][1]["mean_lead_time_steps"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, values, title in zip(
        axes, [pr_aucs, detection_rates, lead_times], ["PR-AUC", "Detection rate", "Mean lead time (steps)"]
    ):
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(title)

    fig.tight_layout()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest_path)
    plt.close(fig)


def archive_previous_latest(out_dir: Path, now: Callable[[], datetime] = datetime.now) -> Path | None:
    latest_dir = out_dir / "latest"
    if not latest_dir.exists():
        return None
    archive_dir = out_dir / "archive" / now().strftime("%Y-%m-%dT%H%M%S")
    shutil.move(str(latest_dir), str(archive_dir))
    return archive_dir


def save_benchmark(
    results: dict[str, tuple[float, dict]], out_dir: Path, now: Callable[[], datetime] = datetime.now
) -> Path:
    archive_previous_latest(out_dir, now=now)
    latest_dir = out_dir / "latest"
    render_comparison_chart(results, latest_dir / "comparison.png")
    return latest_dir
