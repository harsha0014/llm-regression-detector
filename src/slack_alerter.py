import os
import requests
from typing import Optional
from src.models import RunMetadata, RunDiff

def send_slack_alert(
    meta: RunMetadata,
    diff: Optional[RunDiff],
    report_url: str = "https://your-hosted-report-link.html"
) -> bool:
    """Send a Slack alert via incoming webhook. Returns True if successful."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ SLACK_WEBHOOK_URL not set – skipping alert.")
        return False

    if diff is None:
        status = "🟢 First run (baseline established)"
        color = "#36a64f"
    elif diff.critical_triggered:
        status = "🔴 CRITICAL regression detected!"
        color = "#ff0000"
    elif diff.warning_triggered:
        status = "🟡 Warning: regression detected"
        color = "#ffcc00"
    else:
        status = "✅ All good – no regression"
        color = "#36a64f"

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{status}*\nRun: `{meta.run_id}` | Prompt: `{meta.prompt_version}`"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Pass Rate:*\n{meta.pass_rate * 100:.1f}%"},
                {"type": "mrkdwn", "text": f"*DeepEval Score:*\n{meta.avg_summary_similarity:.3f}"},
                {"type": "mrkdwn", "text": f"*Tokens:*\n{meta.total_input_tokens + meta.total_output_tokens}"},
            ]
        }
    ]

    if diff:
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Delta:*\n{diff.pass_rate_delta * 100:+.1f}%"},
                {"type": "mrkdwn", "text": f"*Regressions:*\n{len(diff.regressions)}"},
                {"type": "mrkdwn", "text": f"*Improvements:*\n{len(diff.improvements)}"},
            ]
        })
        if diff.regressions:
            reg_list = "\n".join([f"• `{r.test_case_id}`: {r.predicted_category}" for r in diff.regressions[:3]])
            if len(diff.regressions) > 3:
                reg_list += f"\n... and {len(diff.regressions) - 3} more."
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Regressed Cases:*\n{reg_list}"}
            })

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"<{report_url}|📄 View Full Report>"}
    })

    payload = {
        "attachments": [{
            "color": color,
            "blocks": blocks
        }]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        resp.raise_for_status()
        print("✅ Slack alert sent.")
        return True
    except Exception as e:
        print(f"❌ Failed to send Slack alert: {e}")
        return False