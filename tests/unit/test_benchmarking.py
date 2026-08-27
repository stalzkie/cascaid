from datetime import datetime

from cascaid.benchmarking import archive_previous_latest, render_comparison_chart, save_benchmark


def test_no_archive_when_latest_does_not_exist_yet(tmp_path):
    out_dir = tmp_path / "benchmarks"
    archived = archive_previous_latest(out_dir, now=lambda: datetime(2026, 1, 1, 12, 0, 0))
    assert archived is None
    assert not (out_dir / "archive").exists()


def test_moves_existing_latest_into_timestamped_archive_dir(tmp_path):
    out_dir = tmp_path / "benchmarks"
    latest_dir = out_dir / "latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / "comparison.png").write_bytes(b"fake-png-bytes")

    archived = archive_previous_latest(out_dir, now=lambda: datetime(2026, 1, 1, 12, 30, 0))

    assert archived == out_dir / "archive" / "2026-01-01T123000"
    assert (archived / "comparison.png").read_bytes() == b"fake-png-bytes"
    assert not latest_dir.exists()


def test_render_comparison_chart_writes_a_nonempty_png(tmp_path):
    results = {
        "GNN (real adjacency)": (0.81, {"detection_rate": 0.9, "mean_lead_time_steps": 5.0}),
        "GNN (shuffled adjacency, ablation)": (0.55, {"detection_rate": 0.5, "mean_lead_time_steps": 1.0}),
        "XGBoost (flattened baseline)": (0.70, {"detection_rate": 0.75, "mean_lead_time_steps": 3.0}),
    }
    dest = tmp_path / "comparison.png"

    render_comparison_chart(results, dest)

    assert dest.exists()
    assert dest.stat().st_size > 0


def test_save_benchmark_archives_old_latest_and_writes_new_chart(tmp_path):
    out_dir = tmp_path / "benchmarks"
    old_latest = out_dir / "latest"
    old_latest.mkdir(parents=True)
    (old_latest / "comparison.png").write_bytes(b"old-run")

    results = {
        "GNN (real adjacency)": (0.81, {"detection_rate": 0.9, "mean_lead_time_steps": 5.0}),
    }

    new_latest = save_benchmark(results, out_dir, now=lambda: datetime(2026, 2, 1, 9, 0, 0))

    assert new_latest == out_dir / "latest"
    assert (new_latest / "comparison.png").exists()
    assert (out_dir / "archive" / "2026-02-01T090000" / "comparison.png").read_bytes() == b"old-run"
