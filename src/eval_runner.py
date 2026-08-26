import asyncio
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.models import PromptConfig, TestCase, EvalResult, RunMetadata, RunDiff
from src.llm_client import classify_email
from src.deepeval_metric import score_summary
from src.db import save_run, get_latest_run, get_previous_run  # <-- FIXED: added get_previous_run

async def run_evaluation(
    prompt_version: str,
    dataset_path: Path,
    model: str = "gpt-4o-mini",
    threshold_warning: float = 0.03,
    threshold_critical: float = 0.08,
) -> tuple[RunMetadata, List[EvalResult], Optional[RunDiff]]:
    """
    Load prompt config and golden dataset, run all test cases async,
    compute scores, save to DB, and diff against the previous run.
    Returns (metadata, results, diff) – diff is None if no previous run.
    """
    # 1. Load prompt config
    prompt_file = Path("prompts") / f"{prompt_version}.yaml"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file {prompt_file} not found")
    with open(prompt_file) as f:
        data = yaml.safe_load(f)
    config = PromptConfig(**data)

    # 2. Load dataset
    with open(dataset_path) as f:
        dataset_data = json.load(f)
    test_cases = [TestCase(**tc) for tc in dataset_data["test_cases"]]
    dataset_version = dataset_path.stem  # e.g. "golden_v1"

    # 3. Run all cases asynchronously
    start_time = datetime.now()
    tasks = []
    for tc in test_cases:
        tasks.append(classify_email(tc.input, config))
    raw_outputs = await asyncio.gather(*tasks)

    # 4. Score each result (FIXED: removed duplicate block)
    passed = 0
    total_sim = 0.0
    total_latency = 0.0
    total_in = 0
    total_out = 0
    
    results = []
    for tc, raw in zip(test_cases, raw_outputs):
        cat_match = (raw["category"].strip().lower() == tc.expected_category.strip().lower())
        
        # --- USE DEEPEVAL GEval ---
        deepeval_score = score_summary(
            email=tc.input,
            predicted=raw["summary"],
            expected=tc.expected_summary
        )
        # -------------------------

        if cat_match:
            passed += 1
        total_sim += deepeval_score
        total_latency += raw["latency_ms"]
        total_in += raw["input_tokens"]
        total_out += raw["output_tokens"]

        results.append(EvalResult(
            test_case_id=tc.id,
            predicted_category=raw["category"],
            predicted_summary=raw["summary"],
            expected_category=tc.expected_category,
            expected_summary=tc.expected_summary,
            category_match=cat_match,
            summary_deepeval_score=deepeval_score,
            latency_ms=raw["latency_ms"],
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            total_tokens=raw["total_tokens"],
            error=raw["error"],
        ))

    n = len(results)
    meta = RunMetadata(
        run_id=start_time.strftime("%Y-%m-%dT%H-%M-%S"),
        prompt_version=prompt_version,
        dataset_version=dataset_version,
        model=model,
        started_at=start_time,
        completed_at=datetime.now(),
        total_cases=n,
        passed_cases=passed,
        pass_rate=passed / n if n else 0.0,
        avg_latency_ms=total_latency / n if n else 0.0,
        avg_summary_similarity=total_sim / n if n else 0.0,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        threshold_warning=threshold_warning,
        threshold_critical=threshold_critical,
    )

    # 5. Save to DB
    save_run(meta, results)

    # 6. Diff against previous run
    prev = get_previous_run(meta.run_id)  # <-- Now works because it's imported
    diff = compute_diff(meta, results, prev) if prev else None
    return meta, results, diff


def compute_diff(
    current_meta: RunMetadata,
    current_results: List[EvalResult],
    previous: tuple[RunMetadata, List[EvalResult]]
) -> RunDiff:
    prev_meta, prev_results = previous
    prev_map = {r.test_case_id: r for r in prev_results}
    cur_map = {r.test_case_id: r for r in current_results}

    regressions = []
    improvements = []
    for tc_id, cur_r in cur_map.items():
        prev_r = prev_map.get(tc_id)
        if not prev_r:
            continue  # new test case, ignore for diff
        if prev_r.category_match and not cur_r.category_match:
            regressions.append(cur_r)
        elif not prev_r.category_match and cur_r.category_match:
            improvements.append(cur_r)

    pass_rate_delta = current_meta.pass_rate - prev_meta.pass_rate
    # Per-category delta
    from collections import defaultdict
    prev_cat_passes = defaultdict(int)
    prev_cat_total = defaultdict(int)
    for r in prev_results:
        prev_cat_total[r.expected_category] += 1
        if r.category_match:
            prev_cat_passes[r.expected_category] += 1
    cur_cat_passes = defaultdict(int)
    cur_cat_total = defaultdict(int)
    for r in current_results:
        cur_cat_total[r.expected_category] += 1
        if r.category_match:
            cur_cat_passes[r.expected_category] += 1
    per_cat_delta = {}
    all_cats = set(prev_cat_total.keys()) | set(cur_cat_total.keys())
    for cat in all_cats:
        prev_rate = prev_cat_passes[cat] / prev_cat_total[cat] if prev_cat_total[cat] else 0.0
        cur_rate = cur_cat_passes[cat] / cur_cat_total[cat] if cur_cat_total[cat] else 0.0
        per_cat_delta[cat] = cur_rate - prev_rate

    warning = abs(pass_rate_delta) > current_meta.threshold_warning
    critical = abs(pass_rate_delta) > current_meta.threshold_critical

    return RunDiff(
        baseline_run_id=prev_meta.run_id,
        current_run_id=current_meta.run_id,
        pass_rate_delta=pass_rate_delta,
        per_category_delta=per_cat_delta,
        regressions=regressions,
        improvements=improvements,
        warning_triggered=warning,
        critical_triggered=critical,
    )