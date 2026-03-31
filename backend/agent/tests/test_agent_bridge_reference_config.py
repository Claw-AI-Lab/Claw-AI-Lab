from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_agent_bridge_module():
    module_path = Path(__file__).resolve().parents[2] / "services" / "agent_bridge.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_config_from_template_persists_reference_papers(tmp_path: Path) -> None:
    agent_bridge = _load_agent_bridge_module()
    state = SimpleNamespace(
        runs_base_dir=str(tmp_path / "runs"),
        agent_package_dir=str(tmp_path / "agent"),
    )

    config_path = agent_bridge._generate_config_from_template(
        state,
        project_id="proj-ref-test",
        topic="test topic",
        reference_papers=[
            "1234.56789",
            "/data/papers/local_reference.pdf",
        ],
    )

    content = Path(config_path).read_text(encoding="utf-8")
    assert 'reference_papers:\n    - "1234.56789"\n    - "/data/papers/local_reference.pdf"' in content
    assert "__REFERENCE_PAPERS__" not in content


def test_persist_reference_uploads_saves_pdf_files(tmp_path: Path) -> None:
    agent_bridge = _load_agent_bridge_module()
    project_dir = tmp_path / "project"

    saved_paths = agent_bridge._persist_reference_uploads(
        project_dir,
        [
            {
                "name": "paper one.pdf",
                "contentBase64": base64.b64encode(b"%PDF-1.4 sample").decode("ascii"),
            }
        ],
    )

    assert len(saved_paths) == 1
    saved_path = Path(saved_paths[0])
    assert saved_path.exists()
    assert saved_path.suffix == ".pdf"
    assert saved_path.read_bytes() == b"%PDF-1.4 sample"
