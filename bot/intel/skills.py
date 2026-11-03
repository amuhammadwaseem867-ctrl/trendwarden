"""Import the bundled Trader skill pack's tools as Python modules."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"


def load(skill: str, module: str):
    """load("trader-news", "news") -> the module at .claude/skills/trader-news/news.py"""
    key = f"trader_skill_{skill.replace('-', '_')}_{module}"
    if key in sys.modules:
        return sys.modules[key]
    path = SKILLS_DIR / skill / f"{module}.py"
    if not path.exists():
        raise ImportError(f"{path} not found - the Trader skill pack must be in .claude/skills/")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def script(skill: str, module: str) -> Path:
    return SKILLS_DIR / skill / f"{module}.py"
