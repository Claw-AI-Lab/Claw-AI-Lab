# pyright: basic, reportMissingImports=false, reportUnusedCallResult=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.report import generate_report


class TestReport:
    def test_report_missing_run_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            generate_report(tmp_path / "nonexistent")

    def test_report_no_summary(self, tmp_path: Path):
        with pytest.raises(ValueError, match="pipeline_summary"):
            generate_report(tmp_path)

    def test_report_minimal_run(self, tmp_path: Path):
        (tmp_path / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "rc-test-123",
                    "stages_executed": 23,
                    "stages_done": 23,
                    "stages_blocked": 0,
                    "stages_failed": 0,
                    "final_status": "done",
                    "generated": "2026-03-10T12:00:00Z",
                }
            )
        )
        report = generate_report(tmp_path)
        assert "# ResearchClaw Run Report" in report
        assert "rc-test-123" in report
        assert "✅" in report

    def test_report_with_paper(self, tmp_path: Path):
        (tmp_path / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "test",
                    "stages_executed": 1,
                    "stages_done": 1,
                    "stages_failed": 0,
                    "final_status": "done",
                    "generated": "now",
                }
            )
        )
        draft_dir = tmp_path / "stage-17"
        draft_dir.mkdir()
        (draft_dir / "paper_draft.md").write_text(
            "This is a paper with some words in it."
        )
        report = generate_report(tmp_path)
        assert "Paper" in report
        assert "words" in report

    def test_report_with_citations(self, tmp_path: Path):
        (tmp_path / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "test",
                    "stages_executed": 1,
                    "stages_done": 1,
                    "stages_failed": 0,
                    "final_status": "done",
                    "generated": "now",
                }
            )
        )
        verify_dir = tmp_path / "stage-23"
        verify_dir.mkdir()
        (verify_dir / "verification_report.json").write_text(
            json.dumps(
                {
                    "total_references": 10,
                    "verified_count": 8,
                    "suspicious_count": 1,
                    "hallucinated_count": 1,
                }
            )
        )
        report = generate_report(tmp_path)
        assert "Citations" in report
        assert "8/10" in report
        assert "Suspicious: 1" in report

    def test_report_with_failures(self, tmp_path: Path):
        (tmp_path / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "test",
                    "stages_executed": 5,
                    "stages_done": 3,
                    "stages_failed": 2,
                    "final_status": "failed",
                    "generated": "now",
                }
            )
        )
        report = generate_report(tmp_path)
        assert "❌" in report
        assert "Warnings" in report
        assert "2 stage(s) failed" in report

    def test_report_with_experiment_results(self, tmp_path: Path):
        (tmp_path / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "test",
                    "stages_executed": 1,
                    "stages_done": 1,
                    "stages_failed": 0,
                    "final_status": "done",
                    "generated": "now",
                }
            )
        )
        exp_dir = tmp_path / "stage-12"
        exp_dir.mkdir()
        (exp_dir / "experiment_results.json").write_text(
            json.dumps(
                {
                    "iterations": [{"loss": 0.5}, {"loss": 0.3}],
                    "best_metric": 0.3,
                }
            )
        )
        report = generate_report(tmp_path)
        assert "Experiments" in report
        assert "2 iterations" in report
