import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.models import EvalResult, RunMetadata, RunDiff

DB_PATH = Path("eval_runs/eval_history.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            prompt_version TEXT,
            dataset_version TEXT,
            model TEXT,
            started_at TEXT,
            completed_at TEXT,
            total_cases INTEGER,
            passed_cases INTEGER,
            pass_rate REAL,
            avg_latency_ms REAL,
            avg_summary_similarity REAL,
            total_input_tokens INTEGER,
            total_output_tokens INTEGER,
            threshold_warning REAL,
            threshold_critical REAL,
            results_json TEXT   -- full list of EvalResult as JSON
        )
    """)
    conn.commit()
    conn.close()

def save_run(metadata: RunMetadata, results: List[EvalResult]):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO runs (
            run_id, prompt_version, dataset_version, model,
            started_at, completed_at, total_cases, passed_cases,
            pass_rate, avg_latency_ms, avg_summary_similarity,
            total_input_tokens, total_output_tokens,
            threshold_warning, threshold_critical, results_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metadata.run_id,
        metadata.prompt_version,
        metadata.dataset_version,
        metadata.model,
        metadata.started_at.isoformat(),
        metadata.completed_at.isoformat(),
        metadata.total_cases,
        metadata.passed_cases,
        metadata.pass_rate,
        metadata.avg_latency_ms,
        metadata.avg_summary_similarity,
        metadata.total_input_tokens,
        metadata.total_output_tokens,
        metadata.threshold_warning,
        metadata.threshold_critical,
        json.dumps([r.model_dump() for r in results])
    ))
    conn.commit()
    conn.close()

def get_latest_run() -> Optional[tuple[RunMetadata, List[EvalResult]]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, prompt_version, dataset_version, model,
               started_at, completed_at, total_cases, passed_cases,
               pass_rate, avg_latency_ms, avg_summary_similarity,
               total_input_tokens, total_output_tokens,
               threshold_warning, threshold_critical, results_json
        FROM runs
        ORDER BY started_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    # Parse metadata
    meta = RunMetadata(
        run_id=row[0],
        prompt_version=row[1],
        dataset_version=row[2],
        model=row[3],
        started_at=datetime.fromisoformat(row[4]),
        completed_at=datetime.fromisoformat(row[5]),
        total_cases=row[6],
        passed_cases=row[7],
        pass_rate=row[8],
        avg_latency_ms=row[9],
        avg_summary_similarity=row[10],
        total_input_tokens=row[11],
        total_output_tokens=row[12],
        threshold_warning=row[13],
        threshold_critical=row[14],
    )
    results = [EvalResult(**r) for r in json.loads(row[15])]
    return meta, results



def get_all_runs(limit: int = 10) -> List[RunMetadata]:
    """Fetch the most recent N runs' metadata (without results)."""
    from datetime import datetime
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, prompt_version, dataset_version, model,
               started_at, completed_at, total_cases, passed_cases,
               pass_rate, avg_latency_ms, avg_summary_similarity,
               total_input_tokens, total_output_tokens,
               threshold_warning, threshold_critical
        FROM runs
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    meta_list = []
    for row in rows:
        meta_list.append(RunMetadata(
            run_id=row[0],
            prompt_version=row[1],
            dataset_version=row[2],
            model=row[3],
            started_at=datetime.fromisoformat(row[4]),
            completed_at=datetime.fromisoformat(row[5]),
            total_cases=row[6],
            passed_cases=row[7],
            pass_rate=row[8],
            avg_latency_ms=row[9],
            avg_summary_similarity=row[10],
            total_input_tokens=row[11],
            total_output_tokens=row[12],
            threshold_warning=row[13],
            threshold_critical=row[14],
        ))
    return meta_list

def get_previous_run(current_run_id: str):
    """
    Fetch the run immediately before the given run_id.
    Returns (RunMetadata, List[EvalResult]) or None.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, prompt_version, dataset_version, model,
               started_at, completed_at, total_cases, passed_cases,
               pass_rate, avg_latency_ms, avg_summary_similarity,
               total_input_tokens, total_output_tokens,
               threshold_warning, threshold_critical, results_json
        FROM runs
        WHERE run_id != ?
        ORDER BY started_at DESC
        LIMIT 1
    """, (current_run_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    
    from datetime import datetime
    meta = RunMetadata(
        run_id=row[0],
        prompt_version=row[1],
        dataset_version=row[2],
        model=row[3],
        started_at=datetime.fromisoformat(row[4]),
        completed_at=datetime.fromisoformat(row[5]),
        total_cases=row[6],
        passed_cases=row[7],
        pass_rate=row[8],
        avg_latency_ms=row[9],
        avg_summary_similarity=row[10],
        total_input_tokens=row[11],
        total_output_tokens=row[12],
        threshold_warning=row[13],
        threshold_critical=row[14],
    )
    results = [EvalResult(**r) for r in json.loads(row[15])]
    return meta, results