"""Tests for content quality assessment."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from researchclaw.quality import (
    assess_quality,
    check_strict_quality,
    compute_template_ratio,
    detect_template_content,
)

REAL_ABSTRACT = (
    "We propose a novel method for protein structure prediction using "
    "graph neural networks. Our approach achieves state-of-the-art results "
    "on the CASP14 benchmark with 3.2 GDT-TS improvement over AlphaFold2. "
    "We demonstrate that incorporating side-chain interactions as graph "
    "edges significantly improves local structure accuracy."
)

TEMPLATE_ABSTRACT = (
    "Template abstract: This section will describe the main contributions "
    "of our work. [INSERT your abstract here]. We will discuss the results "
    "in the following sections. Replace this text with your actual content."
)

MIXED_CONTENT = (
    "We propose a novel method for protein structure prediction.\n"
    "[TODO: Add more details about the method]\n"
    "Our experiments show significant improvements over baselines.\n"
    "Template introduction: This section will describe the background."
)

REAL_PAPER_SECTION = (
    "## Introduction\n\n"
    "Recent advances in large language models have demonstrated remarkable "
    "capabilities in natural language understanding and generation. However, "
    "these models often struggle with factual consistency and hallucinate "
    "information. In this work, we address this limitation by introducing "
    "a retrieval-augmented generation framework that grounds model outputs "
    "in verified knowledge sources.\n\n"
    "Our key contributions are:\n"
    "1. A novel attention mechanism for integrating retrieved passages\n"
    "2. A training procedure that incentivizes factual consistency\n"
    "3. Comprehensive evaluation on three benchmark datasets"
)


class TestDetectTemplateContent:
    def test_real_text_no_matches(self):
        matches = detect_template_content(REAL_ABSTRACT)
        assert len(matches) == 0

    def test_template_text_has_matches(self):
        matches = detect_template_content(TEMPLATE_ABSTRACT)
        assert len(matches) >= 3

    def test_detects_insert_placeholder(self):
        text = "The results show [INSERT your results here] improvement."
        matches = detect_template_content(text)
        assert any("Insert placeholder" in m.pattern_desc for m in matches)

    def test_detects_todo_placeholder(self):
        text = "Method description [TODO: complete this section]."
        matches = detect_template_content(text)
        assert any("TODO" in m.pattern_desc for m in matches)

    def test_detects_template_section(self):
        text = "Template introduction: This paper presents our work."
        matches = detect_template_content(text)
        assert any("Template section" in m.pattern_desc for m in matches)

    def test_detects_future_tense_placeholder(self):
        text = "This section will describe the methodology in detail."
        matches = detect_template_content(text)
        assert any("Future-tense" in m.pattern_desc for m in matches)

    def test_detects_lorem_ipsum(self):
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        matches = detect_template_content(text)
        assert any("Lorem ipsum" in m.pattern_desc for m in matches)

    def test_match_has_line_number(self):
        text = "Good line\n[TODO: fix this]\nAnother good line"
        matches = detect_template_content(text)
        assert len(matches) == 1
        assert matches[0].line_number == 2

    def test_real_paper_section_clean(self):
        matches = detect_template_content(REAL_PAPER_SECTION)
        assert len(matches) == 0

    def test_empty_text(self):
        matches = detect_template_content("")
        assert len(matches) == 0


class TestComputeTemplateRatio:
    def test_real_text_low_ratio(self):
        ratio = compute_template_ratio(REAL_ABSTRACT)
        assert ratio < 0.05

    def test_template_text_high_ratio(self):
        ratio = compute_template_ratio(TEMPLATE_ABSTRACT)
        assert ratio > 0.5

    def test_mixed_content_moderate_ratio(self):
        ratio = compute_template_ratio(MIXED_CONTENT)
        assert 0.1 < ratio < 0.9

    def test_empty_text_zero_ratio(self):
        ratio = compute_template_ratio("")
        assert ratio == 0.0

    def test_ratio_bounded_0_1(self):
        ratio = compute_template_ratio(TEMPLATE_ABSTRACT)
        assert 0.0 <= ratio <= 1.0

    def test_real_paper_section_low_ratio(self):
        ratio = compute_template_ratio(REAL_PAPER_SECTION)
        assert ratio < 0.05


class TestAssessQuality:
    def test_report_has_all_fields(self):
        report = assess_quality(REAL_ABSTRACT)
        assert report.total_lines > 0
        assert report.total_chars > 0
        assert isinstance(report.template_ratio, float)
        assert isinstance(report.template_matches, tuple)

    def test_report_to_dict(self):
        report = assess_quality(MIXED_CONTENT)
        d = report.to_dict()
        assert "template_ratio" in d
        assert "template_matches" in d
        assert "has_template_content" in d
        assert "match_count" in d

    def test_report_has_template_flag(self):
        report = assess_quality(TEMPLATE_ABSTRACT)
        assert report.has_template_content is True

        report2 = assess_quality(REAL_ABSTRACT)
        assert report2.has_template_content is False


class TestCheckStrictQuality:
    def test_real_text_passes(self):
        passed, _msg = check_strict_quality(REAL_ABSTRACT)
        assert passed is True

    def test_template_text_fails(self):
        passed, msg = check_strict_quality(TEMPLATE_ABSTRACT)
        assert passed is False
        assert "Template content detected" in msg

    def test_custom_threshold(self):
        passed, _msg = check_strict_quality(TEMPLATE_ABSTRACT, threshold=1.0)
        assert passed is True

    def test_failure_message_includes_examples(self):
        _passed, msg = check_strict_quality(TEMPLATE_ABSTRACT)
        assert "L" in msg
