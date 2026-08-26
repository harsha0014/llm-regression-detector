from src.db import get_all_runs
from typing import Optional

def check_drift(
    current_run_id: str,
    threshold_drift: float = 0.05,  # 5% drop triggers drift warning
    window_size: int = 7
) -> Optional[float]:
    """
    Check if the rolling average pass rate has dropped below the threshold.
    Returns the drift delta if triggered, else None.
    """
    history = get_all_runs(limit=window_size + 1)
    if len(history) < window_size:
        print("ℹ️ Not enough runs for drift detection yet.")
        return None

    # Exclude the current run and compute average of previous N
    previous_runs = [r for r in history if r.run_id != current_run_id][:window_size]
    if len(previous_runs) < window_size:
        return None

    avg_rate = sum(r.pass_rate for r in previous_runs) / len(previous_runs)
    current_rate = next(r.pass_rate for r in history if r.run_id == current_run_id)
    delta = current_rate - avg_rate

    if delta < -threshold_drift:
        return delta
    return None