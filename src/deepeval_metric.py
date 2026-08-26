from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import os

# Global cache for the metric so we don't re-initialize it every time
_metric = None

def get_summary_metric():
    global _metric
    if _metric is None:
        _metric = GEval(
            name="Summary_Relevance",
            criteria="Evaluate if the predicted summary accurately captures the core issue and key details of the expected summary.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,         # The original email
                LLMTestCaseParams.ACTUAL_OUTPUT, # The predicted summary
                LLMTestCaseParams.EXPECTED_OUTPUT # The ground truth summary
            ],
            evaluation_steps=[
                "Check if the predicted summary covers the main intent of the email.",
                "Check if it misses any critical details present in the expected summary.",
                "Assign a score from 0.0 to 1.0, where 1.0 is a perfect match in meaning."
            ],
            model="gpt-4o-mini",  # Cheap judge
            strict_mode=False
        )
    return _metric

def score_summary(email: str, predicted: str, expected: str) -> float:
    """
    Use DeepEval GEval to rate summary relevance.
    Returns a float between 0.0 and 1.0.
    """
    if not predicted or not expected:
        return 0.0
    
    metric = get_summary_metric()
    test_case = LLMTestCase(
        input=email,
        actual_output=predicted,
        expected_output=expected
    )
    
    # DeepEval runs synchronously, we wrap it to avoid blocking the async loop too badly
    metric.measure(test_case)
    # Score is usually 0-1, but let's clamp just in case
    return max(0.0, min(1.0, metric.score))