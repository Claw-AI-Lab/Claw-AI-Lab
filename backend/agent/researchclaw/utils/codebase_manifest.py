"""Generate a compact API manifest for a codebase using AST parsing.

Produces a structured summary (file tree, class/function signatures,
docstrings) that fits in an LLM prompt regardless of codebase size.
Results are cached to ``_manifest.json`` inside the repo directory.
"""
from __future__ import annotations

import ast
import json
import os
import hashlib
from pathlib import Path
from typing import Any


def _signature_from_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a human-readable signature from a function AST node."""
    parts: list[str] = []
    args = node.args

    for i, arg in enumerate(args.args):
        ann = ""
        if arg.annotation:
            try:
                ann = f": {ast.unparse(arg.annotation)}"
            except Exception:
                ann = ": ..."
        default_offset = len(args.args) - len(args.defaults)
        if i >= default_offset:
            default = args.defaults[i - default_offset]
            try:
                dval = ast.unparse(default)
            except Exception:
                dval = "..."
            parts.append(f"{arg.arg}{ann}={dval}")
        else:
            parts.append(f"{arg.arg}{ann}")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kw in args.kwonlyargs:
        ann = ""
        if kw.annotation:
            try:
                ann = f": {ast.unparse(kw.annotation)}"
            except Exception:
                ann = ""
        parts.append(f"{kw.arg}{ann}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            ret = " -> ..."

    return f"({', '.join(parts)}){ret}"


def _get_docstring(node: ast.AST) -> str:
    """Extract the first-line docstring from a class/function node."""
    ds = ast.get_docstring(node)
    if not ds:
        return ""
    first_line = ds.strip().split("\n")[0]
    return first_line[:200]


def _parse_file(filepath: Path, rel_path: str) -> dict[str, Any] | None:
    """Parse a single Python file and extract its API surface."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None

    imports: list[str] = []
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, str]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        elif isinstance(node, (ast.ClassDef,)):
            cls_info: dict[str, Any] = {
                "name": node.name,
                "doc": _get_docstring(node),
                "bases": [],
                "methods": [],
            }
            for base in node.bases:
                try:
                    cls_info["bases"].append(ast.unparse(base))
                except Exception:
                    cls_info["bases"].append("?")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _signature_from_args(item)
                    cls_info["methods"].append({
                        "name": item.name,
                        "sig": sig,
                        "doc": _get_docstring(item),
                    })
            classes.append(cls_info)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _signature_from_args(node)
            functions.append({
                "name": node.name,
                "sig": sig,
                "doc": _get_docstring(node),
            })

    return {
        "file": rel_path,
        "lines": len(source.splitlines()),
        "imports": imports[:30],
        "classes": classes,
        "functions": functions,
    }


_SKIP_DIRS = frozenset((
    ".git", "__pycache__", "node_modules", ".eggs", ".egg-info",
    "docs", "doc", "static", "assets", "images", "figures",
))


def _is_auxiliary_code(rel_path: str) -> bool:
    """Return True for files that are useful but non-core (eval, test, etc.)."""
    parts = rel_path.lower().split("/")
    return any(p in ("eval", "evaluation", "tests", "test", "benchmarks") for p in parts)


def _dir_hash(repo_path: Path) -> str:
    """Quick hash of file mtimes to detect changes."""
    h = hashlib.md5()
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in sorted(dirs) if d not in (".git", "__pycache__", "node_modules", ".eggs")]
        for f in sorted(files):
            if f.endswith((".py", ".yaml", ".yml")):
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp)
                    h.update(f"{fp}:{st.st_mtime_ns}".encode())
                except OSError:
                    pass
    return h.hexdigest()[:12]


def generate_manifest(repo_path: str | Path) -> dict[str, Any]:
    """Generate or load cached API manifest for a codebase."""
    repo_path = Path(repo_path)
    cache_path = repo_path / "_manifest.json"
    current_hash = _dir_hash(repo_path)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("_hash") == current_hash:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    file_entries: list[dict[str, Any]] = []
    file_tree: list[str] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_DIRS]
        for f in sorted(files):
            abs_path = Path(root) / f
            rel = str(abs_path.relative_to(repo_path))
            if rel.startswith("_manifest"):
                continue
            file_tree.append(rel)
            if f.endswith(".py"):
                entry = _parse_file(abs_path, rel)
                if entry:
                    entry["_auxiliary"] = _is_auxiliary_code(rel)
                    file_entries.append(entry)

    # Read README
    readme = ""
    for rn in ("README.md", "readme.md", "README.txt", "README"):
        rp = repo_path / rn
        if rp.is_file():
            try:
                readme = rp.read_text(encoding="utf-8")[:3000]
            except OSError:
                pass
            break

    manifest = {
        "_hash": current_hash,
        "repo_name": repo_path.name,
        "repo_path": str(repo_path),
        "file_tree": file_tree,
        "readme_excerpt": readme,
        "modules": file_entries,
    }

    try:
        cache_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass

    return manifest


def manifest_to_prompt(manifest: dict[str, Any]) -> str:
    """Convert a manifest dict into a compact prompt string for LLM injection."""
    parts: list[str] = []
    repo = manifest.get("repo_name", "unknown")
    repo_path = manifest.get("repo_path", "")

    parts.append(f"## Codebase: `{repo}` (`{repo_path}`)")
    parts.append("")

    readme = manifest.get("readme_excerpt", "")
    if readme:
        readme_clean = _trim_readme(readme)
        if readme_clean:
            parts.append("### README (trimmed)")
            parts.append(f"```\n{readme_clean}\n```")
            parts.append("")

    tree = manifest.get("file_tree", [])
    if tree:
        core_tree = [f for f in tree if not _is_auxiliary_code(f)]
        aux_tree = [f for f in tree if _is_auxiliary_code(f)]
        parts.append(f"### File Tree ({len(core_tree)} core files"
                      + (f", {len(aux_tree)} auxiliary" if aux_tree else "")
                      + ")")
        parts.append("```")
        for f in core_tree:
            parts.append(f"  {f}")
        if aux_tree:
            parts.append(f"  # + {len(aux_tree)} auxiliary files in eval/test dirs")
        parts.append("```")
        parts.append("")

    modules = manifest.get("modules", [])
    core_mods = [m for m in modules if not m.get("_auxiliary")]
    aux_mods = [m for m in modules if m.get("_auxiliary")]
    if core_mods:
        parts.append("### API Reference")
        for mod in core_mods:
            fpath = mod["file"]
            lines = mod.get("lines", 0)
            parts.append(f"\n**`{fpath}`** ({lines} lines)")

            for cls in mod.get("classes", []):
                bases = ", ".join(cls.get("bases", []))
                base_str = f"({bases})" if bases else ""
                parts.append(f"  class **{cls['name']}**{base_str}")
                if cls.get("doc"):
                    parts.append(f"    {cls['doc']}")
                for m in cls.get("methods", []):
                    parts.append(f"    def {m['name']}{m['sig']}")
                    if m.get("doc"):
                        parts.append(f"      {m['doc']}")

            for fn in mod.get("functions", []):
                parts.append(f"  def **{fn['name']}**{fn['sig']}")
                if fn.get("doc"):
                    parts.append(f"    {fn['doc']}")

    if aux_mods:
        parts.append(f"\n*Auxiliary scripts ({len(aux_mods)} files in eval/test): "
                      + ", ".join(m["file"] for m in aux_mods) + "*")

    parts.append(f"\n**Usage**: `sys.path.insert(0, '{repo_path}')` then import modules.")
    return "\n".join(parts)


def _trim_readme(readme: str) -> str:
    """Strip HTML tags, badges, image links, and boilerplate from README."""
    import re
    lines = readme.splitlines()
    trimmed: list[str] = []
    skip_section = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^</?(?:div|a|img|h[1-6]|p|b|br|hr|span)\b', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\[!\[', stripped) or re.match(r'^!\[', stripped):
            continue
        if re.match(r'^\[.+\]\(https?://', stripped) and len(stripped) < 300:
            if "arxiv" not in stripped.lower() and "paper" not in stripped.lower():
                continue
        if stripped.startswith("## ") and any(kw in stripped.lower() for kw in
                ("install", "run", "setup", "getting started", "usage",
                 "prerequisit", "requirement", "depend", "quick start")):
            skip_section = True
            continue
        if re.match(r'^#{1,3} ', stripped) and skip_section:
            skip_section = False
        if skip_section:
            continue
        if not stripped:
            if trimmed and not trimmed[-1].strip():
                continue
        trimmed.append(line)
    text = re.sub(r'<[^>]+>', '', "\n".join(trimmed)).strip()
    return text[:1500]
