from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class PromptConfig(BaseModel):
    version: str
    timestamp: str
    system_prompt: str
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list)
    # e.g. [{"email": "...", "category": "...", "summary": "..."}]

class TestCase(BaseModel):
    id: str                     # e.g. "tc_001"
    input: str                  # email text
    expected_category: str
    expected_summary: str
    difficulty: str             # "easy", "medium", "hard"
    notes: Optional[str] = None

class EvalResult(BaseModel):
    test_case_id: str
    predicted_category: str
    predicted_summary: str
    expected_category: str
    expected_summary: str
    category_match: bool
    summary_deepeval_score: float = 0.0  # <-- NEW FIELD (0-1 from GEval)
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    error: Optional[str] = None

class RunMetadata(BaseModel):
    run_id: str                 # timestamp based, e.g. "2025-01-15T10-30-00"
    prompt_version: str
    dataset_version: str        # from golden dataset file name
    model: str                  # "gpt-4o-mini"
    started_at: datetime
    completed_at: datetime
    total_cases: int
    passed_cases: int           # category_match == True
    pass_rate: float
    avg_latency_ms: float
    avg_summary_similarity: float
    total_input_tokens: int
    total_output_tokens: int
    threshold_warning: float    # 0.03 (3%) – configurable
    threshold_critical: float   # 0.08 (8%)

class RunDiff(BaseModel):
    baseline_run_id: Optional[str]
    current_run_id: str
    pass_rate_delta: float
    per_category_delta: Dict[str, float]
    regressions: List[EvalResult]   # previously passed, now failed
    improvements: List[EvalResult]  # previously failed, now passed
    warning_triggered: bool
    critical_triggered: bool