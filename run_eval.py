import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from src.eval_runner import run_evaluation
from src.report_generator import generate_html_report
from src.slack_alerter import send_slack_alert
from src.drift_detector import check_drift

load_dotenv()

async def main():
    # Configurable parameters
    prompt_version = "v1.1"  # Change this to test different prompts
    dataset_path = Path("golden_dataset/golden_v1.json")
    threshold_warning = 0.03
    threshold_critical = 0.08
    drift_threshold = 0.05   # 5% drop triggers drift warning

    print(f"🚀 Running eval with prompt {prompt_version} ...")
    meta, results, diff = await run_evaluation(
        prompt_version=prompt_version,
        dataset_path=dataset_path,
        threshold_warning=threshold_warning,
        threshold_critical=threshold_critical,
    )

    # --- Phase 4 Features (Reports, Slack, Drift) ---
    # 1. Generate HTML report
    report_path = generate_html_report(meta, results, diff)
    print(f"📄 Report saved: {report_path.absolute()}")

    # 2. Check for slow drift
    drift_delta = check_drift(meta.run_id, threshold_drift=drift_threshold)
    if drift_delta is not None:
        print(f"🌊 SLOW DRIFT detected: pass rate dropped {drift_delta * 100:.1f}% vs rolling average.")

    # 3. Send Slack alert (set SLACK_WEBHOOK_URL in .env to enable)
    report_url = "https://your-hosted-report-link.html"  # Replace if hosting
    send_slack_alert(meta, diff, report_url=report_url)

    # --- Console Summary ---
    print(f"\n✅ Run ID: {meta.run_id}")
    print(f"   Pass rate: {meta.pass_rate:.2%}")
    print(f"   Avg DeepEval Summary Score: {meta.avg_summary_similarity:.3f}")
    print(f"   Avg latency: {meta.avg_latency_ms:.1f} ms")

    if diff:
        print(f"\nDiff vs previous run ({diff.baseline_run_id}):")
        print(f"  Pass rate delta: {diff.pass_rate_delta:+.2%}")
        print(f"  Regressions: {len(diff.regressions)}")
        print(f"  Improvements: {len(diff.improvements)}")
        if diff.critical_triggered:
            print("  ❌ CRITICAL regression detected!")
        elif diff.warning_triggered:
            print("  ⚠️  Warning regression detected.")
        else:
            print("  ✅ No significant regression.")
    else:
        print("\nNo previous run found – this is the baseline.")

    # --- Phase 5 CI/CD Exit Logic ---
    # This is the part you asked about – it's perfectly placed here.
    if diff and diff.critical_triggered:
        print("❌ Critical regression detected – failing CI build.")
        sys.exit(1)   # Blocks merge in GitHub Actions
    else:
        print("✅ Build passed.")
        sys.exit(0)   # Allows merge

if __name__ == "__main__":
    asyncio.run(main())