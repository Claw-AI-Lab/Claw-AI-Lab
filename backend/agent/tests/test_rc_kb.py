from __future__ import annotations

import json
from pathlib import Path

import yaml

from researchclaw.knowledge.base import (
    KB_CATEGORY_MAP,
    KBEntry,
    _markdown_frontmatter,
    _obsidian_enhancements,
    generate_weekly_report,
    write_kb_entry,
    write_stage_to_kb,
)


def _kb_root(tmp_path: Path) -> Path:
    return tmp_path / "kb"


def test_kb_entry_dataclass_creation():
    entry = KBEntry(
        category="findings",
        entry_id="e1",
        title="T",
        content="C",
        source_stage="01-goal_define",
        run_id="run1",
    )
    assert entry.category == "findings"
    assert entry.entry_id == "e1"
    assert entry.run_id == "run1"


def test_write_kb_entry_creates_expected_file_path(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    entry = KBEntry("questions", "q-1", "Q", "Body", "01-goal_define", "run-a")
    path = write_kb_entry(kb_root, entry)
    assert path == kb_root / "questions" / "q-1.md"
    assert path.exists()


def test_write_kb_entry_includes_frontmatter_markers(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    entry = KBEntry("findings", "f-1", "Finding", "Body", "14-result_analysis", "run-a")
    text = write_kb_entry(kb_root, entry).read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "\n---\n" in text


def test_write_kb_entry_markdown_backend_has_no_obsidian_extras(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    entry = KBEntry(
        "questions",
        "q-2",
        "Question",
        "Body",
        "01-goal_define",
        "run-a",
        tags=["hypothesis"],
        links=["run-run-a"],
    )
    text = write_kb_entry(kb_root, entry, backend="markdown").read_text(
        encoding="utf-8"
    )
    assert "[[run-run-a]]" not in text
    assert "#hypothesis" not in text


def test_write_kb_entry_obsidian_backend_includes_tags_and_wikilinks(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    entry = KBEntry(
        "questions",
        "q-3",
        "Question",
        "Body",
        "01-goal_define",
        "run-a",
        tags=["hypothesis", "q1"],
        links=["run-run-a", "topic-a"],
    )
    text = write_kb_entry(kb_root, entry, backend="obsidian").read_text(
        encoding="utf-8"
    )
    assert "#hypothesis #q1" in text
    assert "Related: [[run-run-a]], [[topic-a]]" in text


def test_markdown_frontmatter_output_format_and_fields():
    entry = KBEntry(
        "reviews",
        "r-1",
        "Report",
        "Body",
        "report",
        "run-x",
        tags=["weekly"],
        evidence_refs=["stage-01/goal.md"],
    )
    fm = _markdown_frontmatter(entry)
    assert fm.startswith("---\n")
    assert fm.endswith("\n---\n")
    parsed = yaml.safe_load(fm.split("---\n", 1)[1].rsplit("\n---\n", 1)[0])
    assert parsed["id"] == "r-1"
    assert parsed["title"] == "Report"
    assert parsed["stage"] == "report"
    assert parsed["run_id"] == "run-x"
    assert parsed["tags"] == ["weekly"]
    assert parsed["evidence"] == ["stage-01/goal.md"]


def test_obsidian_enhancements_with_tags_and_links():
    entry = KBEntry(
        "findings",
        "f-2",
        "Finding",
        "Body",
        "14-result_analysis",
        "run-z",
        tags=["a", "b"],
        links=["run-z", "result-node"],
    )
    enh = _obsidian_enhancements(entry)
    assert "#a #b" in enh
    assert "Related: [[run-z]], [[result-node]]" in enh


def test_obsidian_enhancements_with_no_tags_or_links_returns_empty():
    entry = KBEntry("findings", "f-3", "Finding", "Body", "14-result_analysis", "run-z")
    assert _obsidian_enhancements(entry) == ""


def test_kb_category_map_has_exactly_22_stage_entries():
    assert len(KB_CATEGORY_MAP) == 22
    assert set(KB_CATEGORY_MAP) == set(range(1, 23))


def test_kb_category_map_values_are_valid_categories():
    valid = {
        "questions",
        "literature",
        "experiments",
        "findings",
        "decisions",
        "reviews",
    }
    assert set(KB_CATEGORY_MAP.values()).issubset(valid)


def test_write_stage_to_kb_places_entry_in_mapped_category(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    stage_dir = tmp_path / "stage-10"
    stage_dir.mkdir()
    (stage_dir / "run.md").write_text("exp content", encoding="utf-8")
    paths = write_stage_to_kb(
        kb_root, 10, "experiment_cycle", "run-1", ["run.md"], stage_dir
    )
    assert len(paths) == 1
    assert paths[0].parent.name == "experiments"


def test_write_stage_to_kb_reads_artifact_file_contents(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    stage_dir = tmp_path / "stage-04"
    stage_dir.mkdir()
    (stage_dir / "lit.md").write_text("paper A\npaper B", encoding="utf-8")
    path = write_stage_to_kb(
        kb_root, 4, "literature_search", "run-1", ["lit.md"], stage_dir
    )[0]
    text = path.read_text(encoding="utf-8")
    assert "paper A" in text
    assert "stage-04/lit.md" in text


def test_write_stage_to_kb_handles_missing_artifacts_gracefully(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    stage_dir = tmp_path / "stage-05"
    stage_dir.mkdir()
    path = write_stage_to_kb(
        kb_root, 5, "literature_extract", "run-2", ["missing.md"], stage_dir
    )[0]
    text = path.read_text(encoding="utf-8")
    assert "Stage 05 (literature_extract) completed" in text


def test_write_stage_to_kb_truncates_large_artifact_content(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    stage_dir = tmp_path / "stage-12"
    stage_dir.mkdir()
    large_text = "x" * 6000
    (stage_dir / "big.txt").write_text(large_text, encoding="utf-8")
    path = write_stage_to_kb(
        kb_root, 12, "experiment_implement", "run-3", ["big.txt"], stage_dir
    )[0]
    text = path.read_text(encoding="utf-8")
    assert "... (truncated, see full artifact)" in text
    assert text.count("x") >= 5000


def test_write_stage_to_kb_directory_artifact_records_listing(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    stage_dir = tmp_path / "stage-13"
    artifact_dir = stage_dir / "outputs"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "a.txt").write_text("a", encoding="utf-8")
    (artifact_dir / "b.txt").write_text("b", encoding="utf-8")
    path = write_stage_to_kb(
        kb_root, 13, "experiment_execute", "run-4", ["outputs/"], stage_dir
    )[0]
    text = path.read_text(encoding="utf-8")
    assert "Directory with 2 files: a.txt, b.txt" in text
    assert "stage-13/outputs/" in text


def test_generate_weekly_report_creates_file_in_reviews_category(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps({"run_id": "run-a", "stages_executed": 10, "stages_done": 10}),
        encoding="utf-8",
    )
    path = generate_weekly_report(kb_root, [run_dir], week_label="2026-W10")
    assert path.parent.name == "reviews"
    assert path.name == "weekly-report-2026-W10.md"


def test_generate_weekly_report_with_empty_run_dirs(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    path = generate_weekly_report(kb_root, [], week_label="2026-W11")
    text = path.read_text(encoding="utf-8")
    assert "Pipeline runs: 0" in text
    assert "Success rate: N/A" in text


def test_generate_weekly_report_aggregates_statistics_correctly(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    run1 = tmp_path / "run-1"
    run2 = tmp_path / "run-2"
    run1.mkdir()
    run2.mkdir()
    (run1 / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "stages_executed": 20,
                "stages_done": 18,
                "stages_failed": 1,
                "stages_blocked": 1,
                "final_status": "failed",
            }
        ),
        encoding="utf-8",
    )
    (run2 / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-2",
                "stages_executed": 10,
                "stages_done": 10,
                "stages_failed": 0,
                "stages_blocked": 0,
                "final_status": "done",
            }
        ),
        encoding="utf-8",
    )
    report = generate_weekly_report(kb_root, [run1, run2], week_label="2026-W12")
    text = report.read_text(encoding="utf-8")
    assert "Pipeline runs: 2" in text
    assert "Stages executed: 30" in text
    assert "Stages completed: 28" in text
    assert "Stages failed: 1" in text
    assert "Stages blocked (gate): 1" in text
    assert "Success rate: 93.3%" in text


def test_generate_weekly_report_ignores_missing_summary_files(tmp_path: Path):
    kb_root = _kb_root(tmp_path)
    run_ok = tmp_path / "run-ok"
    run_empty = tmp_path / "run-empty"
    run_ok.mkdir()
    run_empty.mkdir()
    (run_ok / "pipeline_summary.json").write_text(
        json.dumps({"run_id": "run-ok", "stages_executed": 5, "stages_done": 5}),
        encoding="utf-8",
    )
    report = generate_weekly_report(kb_root, [run_ok, run_empty], week_label="2026-W13")
    text = report.read_text(encoding="utf-8")
    assert "Pipeline runs: 1" in text
