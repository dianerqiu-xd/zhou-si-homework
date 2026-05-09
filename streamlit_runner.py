from __future__ import annotations

import os
import runpy
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zhou_si_mplconfig"))
APP_MODULE_NAMES = {
    "core",
    "image_color_interpolation",
    "image_filtering",
    "vibe_feature_core",
}


def _clear_app_modules() -> None:
    """Avoid cross-page collisions between each homework app's local src package."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src.") or name in APP_MODULE_NAMES:
            del sys.modules[name]


@contextmanager
def _app_context(app_dir: Path):
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    _clear_app_modules()
    os.chdir(app_dir)
    sys.path.insert(0, str(app_dir))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def run_app(folder: str, entrypoint: str) -> None:
    app_dir = ROOT / folder
    script = app_dir / entrypoint
    if not script.exists():
        raise FileNotFoundError(f"Missing Streamlit app entrypoint: {script}")
    with _app_context(app_dir):
        runpy.run_path(str(script), run_name="__main__")
