"""Tests for researchclaw.hardware — GPU detection & metric filtering."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.hardware import (
    HardwareProfile,
    _detect_mps,
    _detect_nvidia,
    detect_hardware,
    ensure_torch_available,
    is_metric_name,
)


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

class TestHardwareProfile:
    def test_to_dict(self):
        hp = HardwareProfile(
            has_gpu=True, gpu_type="cuda", gpu_name="RTX 4090",
            vram_mb=24564, tier="high", warning="",
        )
        d = hp.to_dict()
        assert d["has_gpu"] is True
        assert d["gpu_type"] == "cuda"
        assert d["vram_mb"] == 24564

    def test_cpu_only_profile(self):
        hp = HardwareProfile(
            has_gpu=False, gpu_type="cpu", gpu_name="CPU only",
            vram_mb=None, tier="cpu_only", warning="No GPU",
        )
        assert hp.tier == "cpu_only"
        assert hp.warning == "No GPU"


# ---------------------------------------------------------------------------
# NVIDIA detection
# ---------------------------------------------------------------------------

class TestDetectNvidia:
    def test_high_vram_nvidia(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090, 24564\n"

        with patch("researchclaw.hardware.subprocess.run", return_value=mock_result):
            profile = _detect_nvidia()

        assert profile is not None
        assert profile.has_gpu is True
        assert profile.gpu_type == "cuda"
        assert profile.gpu_name == "NVIDIA GeForce RTX 4090"
        assert profile.vram_mb == 24564
        assert profile.tier == "high"
        assert profile.warning == ""

    def test_low_vram_nvidia(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce GTX 1650, 4096\n"

        with patch("researchclaw.hardware.subprocess.run", return_value=mock_result):
            profile = _detect_nvidia()

        assert profile is not None
        assert profile.tier == "limited"
        assert "limited memory" in profile.warning

    def test_nvidia_smi_not_found(self):
        with patch(
            "researchclaw.hardware.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _detect_nvidia() is None

    def test_nvidia_smi_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("researchclaw.hardware.subprocess.run", return_value=mock_result):
            assert _detect_nvidia() is None

    def test_nvidia_smi_timeout(self):
        with patch(
            "researchclaw.hardware.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            assert _detect_nvidia() is None


# ---------------------------------------------------------------------------
# MPS detection
# ---------------------------------------------------------------------------

class TestDetectMPS:
    def test_apple_silicon(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Apple M3 Pro\n"

        with (
            patch("researchclaw.hardware.platform.system", return_value="Darwin"),
            patch("researchclaw.hardware.platform.machine", return_value="arm64"),
            patch("researchclaw.hardware.subprocess.run", return_value=mock_result),
        ):
            profile = _detect_mps()

        assert profile is not None
        assert profile.has_gpu is True
        assert profile.gpu_type == "mps"
        assert profile.gpu_name == "Apple M3 Pro"
        assert profile.tier == "limited"
        assert "MPS" in profile.warning

    def test_non_darwin(self):
        with patch("researchclaw.hardware.platform.system", return_value="Linux"):
            assert _detect_mps() is None

    def test_intel_mac(self):
        with (
            patch("researchclaw.hardware.platform.system", return_value="Darwin"),
            patch("researchclaw.hardware.platform.machine", return_value="x86_64"),
        ):
            assert _detect_mps() is None


# ---------------------------------------------------------------------------
# detect_hardware (integration)
# ---------------------------------------------------------------------------

class TestDetectHardware:
    def test_falls_back_to_cpu(self):
        with (
            patch("researchclaw.hardware._detect_nvidia", return_value=None),
            patch("researchclaw.hardware._detect_mps", return_value=None),
        ):
            profile = detect_hardware()

        assert profile.has_gpu is False
        assert profile.gpu_type == "cpu"
        assert profile.tier == "cpu_only"
        assert "No GPU" in profile.warning

    def test_nvidia_takes_priority(self):
        nvidia_profile = HardwareProfile(
            has_gpu=True, gpu_type="cuda", gpu_name="RTX 4090",
            vram_mb=24564, tier="high", warning="",
        )
        mps_profile = HardwareProfile(
            has_gpu=True, gpu_type="mps", gpu_name="M3",
            vram_mb=None, tier="limited", warning="MPS",
        )
        with (
            patch("researchclaw.hardware._detect_nvidia", return_value=nvidia_profile),
            patch("researchclaw.hardware._detect_mps", return_value=mps_profile),
        ):
            profile = detect_hardware()

        assert profile.gpu_type == "cuda"


# ---------------------------------------------------------------------------
# ensure_torch_available
# ---------------------------------------------------------------------------

class TestEnsureTorchAvailable:
    def test_already_installed(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "2.3.0\n"

        with patch("researchclaw.hardware.subprocess.run", return_value=mock_result):
            assert ensure_torch_available("/usr/bin/python3", "cuda") is True

    def test_cpu_only_skips_install(self):
        mock_check = MagicMock()
        mock_check.returncode = 1  # not installed
        mock_check.stdout = ""

        with patch("researchclaw.hardware.subprocess.run", return_value=mock_check):
            assert ensure_torch_available("/usr/bin/python3", "cpu") is False

    def test_install_succeeds(self):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            mock = MagicMock()
            if call_count["n"] == 1:
                mock.returncode = 1  # import check fails
                mock.stdout = ""
            else:
                mock.returncode = 0  # pip install succeeds
                mock.stdout = ""
            return mock

        with patch("researchclaw.hardware.subprocess.run", side_effect=side_effect):
            assert ensure_torch_available("/usr/bin/python3", "cuda") is True

    def test_install_fails(self):
        mock = MagicMock()
        mock.returncode = 1
        mock.stdout = ""
        mock.stderr = "ERROR: Could not install"

        with patch("researchclaw.hardware.subprocess.run", return_value=mock):
            assert ensure_torch_available("/usr/bin/python3", "mps") is False

    def test_python_not_found(self):
        with patch(
            "researchclaw.hardware.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert ensure_torch_available("/nonexistent/python3", "cuda") is False


# ---------------------------------------------------------------------------
# is_metric_name
# ---------------------------------------------------------------------------

class TestIsMetricName:
    def test_valid_metrics(self):
        assert is_metric_name("loss") is True
        assert is_metric_name("primary_metric") is True
        assert is_metric_name("UCB (Stochastic) cumulative_regret") is True
        assert is_metric_name("accuracy") is True
        assert is_metric_name("f1_score") is True

    def test_log_lines_filtered(self):
        assert is_metric_name("Running experiments for support set size") is False
        assert is_metric_name("Loading model weights") is False
        assert is_metric_name("Training epoch 5") is False
        assert is_metric_name("Evaluating on test set") is False
        assert is_metric_name("Processing batch") is False
        assert is_metric_name("Initializing optimizer") is False

    def test_too_many_words_filtered(self):
        assert is_metric_name("this is a very long name that has many words") is False

    def test_short_names_pass(self):
        assert is_metric_name("val_loss") is True
        assert is_metric_name("test accuracy score") is True
