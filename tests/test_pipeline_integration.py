"""End-to-end wiring test: exercises cli.main() -> pipeline.run_experiment()
exactly as a user invoking `python main.py --quick` would, so a break in the
CLI/pipeline glue (as opposed to the individual algorithm modules, which
have their own unit tests) fails `pytest` instead of only surfacing on a
manual run."""

import json

from inel.cli import main


def test_quick_cli_run_produces_expected_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    rc = main([
        "--quick",
        "--data-dir", str(data_dir),
        "--results-dir", str(results_dir),
        "--no-plots",
    ])
    assert rc == 0

    for name in ("baseline.json", "ea2007.json", "neat.json", "all_metrics.json"):
        assert (results_dir / name).exists(), f"missing {name}"

    metrics = json.loads((results_dir / "all_metrics.json").read_text())
    assert set(metrics) == {"baseline", "ea2007", "neat"}
    for cond in metrics.values():
        for key in ("RA", "RA_mean", "FR", "FR_mean", "FT", "FT_mean", "EC", "EC_runs"):
            assert key in cond


def test_quick_cli_run_is_cached_on_second_invocation(tmp_path):
    """load_or_run should skip recomputation once results exist on disk."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    args = ["--quick", "--data-dir", str(data_dir), "--results-dir", str(results_dir), "--no-plots"]

    assert main(args) == 0
    first = (results_dir / "baseline.json").read_text()

    assert main(args) == 0
    second = (results_dir / "baseline.json").read_text()
    assert first == second
