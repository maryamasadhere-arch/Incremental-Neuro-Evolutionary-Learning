from inel.metrics import compute_metrics


def _episode_matrix(rows):
    """rows: list of per-episode accuracy lists (a single run)."""
    return [rows]


def test_ra_is_final_episode_accuracy_per_task():
    runs = _episode_matrix([
        [0.95],
        [0.60, 0.96],
        [0.55, 0.70, 0.97],
    ])
    m = compute_metrics(runs, "x", N=3)
    assert m["RA"] == [55.0, 70.0, 97.0]


def test_fr_excludes_final_task_and_uses_peak_minus_final():
    # Task 0: peak 0.95 (episode 0), ends at 0.55 -> FR = 40%
    # Task 1: peak 0.80 (episode 1), ends at 0.70 -> FR = 10%
    # Task 2 (final task): excluded from FR entirely
    runs = _episode_matrix([
        [0.95],
        [0.60, 0.80],
        [0.55, 0.70, 0.97],
    ])
    m = compute_metrics(runs, "x", N=3)
    assert m["FR"] == [40.0, 10.0]


def test_ec_requires_separate_prior_and_current_thresholds():
    """Report Sec. 2.5: RA > 70% on ALL prior tasks AND accuracy > 85% on
    the CURRENT task - two different thresholds, not one shared 70%."""
    # Episode 0: current-task acc 0.80 -> fails the 85% current-task bar
    # even though there are no prior tasks to fail on.
    runs = _episode_matrix([
        [0.80],
    ])
    m = compute_metrics(runs, "x", N=1, ec_prior_threshold=0.70, ec_current_threshold=0.85)
    assert m["EC"] == 0.0


def test_ec_passes_when_current_task_clears_the_higher_bar():
    runs = _episode_matrix([
        [0.90],
    ])
    m = compute_metrics(runs, "x", N=1, ec_prior_threshold=0.70, ec_current_threshold=0.85)
    assert m["EC"] == 1.0


def test_ec_stops_at_first_failure_not_last_success():
    """A ceiling that could be 'regained' after a mid-sequence collapse
    isn't a stable ceiling - EC must reflect the longest *contiguous*
    successful prefix, not the last episode that happened to pass."""
    runs = _episode_matrix([
        [0.90],                    # ep0: pass -> ceiling so far = 1
        [0.50, 0.90],              # ep1: prior task collapsed -> fail, stop
        [0.95, 0.95, 0.90],        # ep2: passes again, but must NOT count
    ])
    m = compute_metrics(runs, "x", N=3, ec_prior_threshold=0.70, ec_current_threshold=0.85)
    assert m["EC"] == 1.0


def test_ft_excludes_first_task_and_measures_relative_to_chance():
    # Task 1 (index 1) accuracy when first trained (episode 1) is 0.80 -> FT = 30
    runs = _episode_matrix([
        [0.95],
        [0.60, 0.80],
    ])
    m = compute_metrics(runs, "x", N=2)
    assert m["FT"] == [30.0]
