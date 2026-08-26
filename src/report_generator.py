import json
from pathlib import Path
from typing import List, Optional
from src.models import RunMetadata, EvalResult, RunDiff
from src.db import get_all_runs, get_previous_run

def generate_html_report(
    current_meta: RunMetadata,
    results: List[EvalResult],
    diff: Optional[RunDiff],
    output_path: Path = Path("report.html")
) -> Path:
    """Generate a standalone HTML report with DeepEval scores and regression table."""
    # Fetch historical data for trend chart
    history = get_all_runs(limit=10)
    trend_data = [{"run": r.run_id[:10], "rate": r.pass_rate * 100} for r in history]

    # Prepare regression rows
    regression_rows = ""
    if diff and diff.regressions:
        # Find the previous results for these cases
        prev_data = get_previous_run(current_meta.run_id)
        prev_map = {r.test_case_id: r for r in prev_data[1]} if prev_data else {}
        for reg in diff.regressions:
            old = prev_map.get(reg.test_case_id)
            regression_rows += f"""
            <tr class="regression">
                <td><strong>{reg.test_case_id}</strong></td>
                <td style="background:#ffdddd">{old.predicted_category if old else 'N/A'}</td>
                <td style="background:#ddffdd">{reg.predicted_category}</td>
                <td style="background:#ffdddd">{old.predicted_summary if old else 'N/A'}</td>
                <td style="background:#ddffdd">{reg.predicted_summary}</td>
                <td style="background:#ffdddd">{old.summary_deepeval_score:.2f if old else 'N/A'}</td>
                <td style="background:#ddffdd">{reg.summary_deepeval_score:.2f}</td>
            </tr>
            """

    # Build the HTML – FIX: handle diff being None
    diff_section = ""
    if diff:
        diff_section = f"""
        <h2>📋 Diff vs Baseline</h2>
        <p><strong>Baseline Run:</strong> {diff.baseline_run_id}</p>
        <div class="scorecard">
            <div class="card"><h3>Pass Rate Δ</h3><div class="value {'pass' if diff.pass_rate_delta >= 0 else 'fail'}">{diff.pass_rate_delta * 100:+.1f}%</div></div>
            <div class="card"><h3>Regressions</h3><div class="value fail">{len(diff.regressions)}</div></div>
            <div class="card"><h3>Improvements</h3><div class="value pass">{len(diff.improvements)}</div></div>
        </div>
        {f'<p class="fail"><strong>❌ CRITICAL</strong> regression detected!</p>' if diff.critical_triggered else ''}
        {f'<p class="warn"><strong>⚠️ WARNING</strong> threshold exceeded.</p>' if diff.warning_triggered and not diff.critical_triggered else ''}
        {f'<p class="pass"><strong>✅ No significant regression.</strong></p>' if not diff.warning_triggered and not diff.critical_triggered else ''}
        """
    else:
        diff_section = """
        <h2>📋 First Run</h2>
        <p><strong>This is the first run – no baseline to compare against.</strong></p>
        <p>Future runs will show diff metrics here.</p>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Regression Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; }}
            .scorecard {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
            .card {{ background: #e9ecef; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px; }}
            .card h3 {{ margin: 0; color: #495057; }}
            .card .value {{ font-size: 28px; font-weight: bold; }}
            .pass {{ color: green; }} .fail {{ color: red; }} .warn {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .regression {{ background-color: #fff0f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Model Regression Report</h1>
            <p><strong>Run ID:</strong> {current_meta.run_id}</p>
            <p><strong>Prompt:</strong> {current_meta.prompt_version} | <strong>Dataset:</strong> {current_meta.dataset_version} | <strong>Model:</strong> {current_meta.model}</p>
            <p><strong>Completed:</strong> {current_meta.completed_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="scorecard">
                <div class="card"><h3>Pass Rate</h3><div class="value">{current_meta.pass_rate * 100:.1f}%</div></div>
                <div class="card"><h3>DeepEval Score</h3><div class="value">{current_meta.avg_summary_similarity:.3f}</div></div>
                <div class="card"><h3>Avg Latency</h3><div class="value">{current_meta.avg_latency_ms:.0f} ms</div></div>
                <div class="card"><h3>Total Tokens</h3><div class="value">{current_meta.total_input_tokens + current_meta.total_output_tokens}</div></div>
            </div>

            <h2>📊 Trend (Last {len(trend_data)} Runs)</h2>
            <canvas id="trendChart" width="800" height="300"></canvas>
            <script>
                const ctx = document.getElementById('trendChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: {json.dumps([d['run'] for d in trend_data])},
                        datasets: [{{
                            label: 'Pass Rate (%)',
                            data: {json.dumps([d['rate'] for d in trend_data])},
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{ y: {{ min: 0, max: 100 }} }}
                    }}
                }});
            </script>

            {diff_section}

            <h2>🐛 Regressed Cases (Old vs New)</h2>
            <table>
                <tr><th>Case ID</th><th>Old Category</th><th>New Category</th><th>Old Summary</th><th>New Summary</th><th>Old DeepEval</th><th>New DeepEval</th></tr>
                {regression_rows if regression_rows else '<tr><td colspan="7">🎉 No regressions found!</td></tr>'}
            </table>
        </div>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path