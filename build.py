"""Build script."""

import shutil
from distutils import log as distutils_log
from pathlib import Path
from typing import Any, Dict

import skbuild
import skbuild.constants

__all__ = ("build",)


def build(setup_kwargs: Dict[str, Any]) -> None:
    """Build C-extensions."""
    skbuild.setup(**setup_kwargs, script_args=["build_ext", "build"])

    src_dir = Path(skbuild.constants.CMAKE_INSTALL_DIR()) / "vendor" / "whisper.cpp"
    print(src_dir)
    dest_dir = Path("whisper_cpp_python")
    print(dest_dir)

    # Delete C-extensions copied in previous runs, just in case.
    remove_files(dest_dir, "**/*.so")

    # Copy built C-extensions back to the project.
    copy_files(src_dir, dest_dir, "**/*.so")


def remove_files(target_dir: Path, pattern: str) -> None:
    """Delete files matched with a glob pattern in a directory tree."""
    for path in target_dir.glob(pattern):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        distutils_log.info(f"removed {path}")


def copy_files(src_dir: Path, dest_dir: Path, pattern: str) -> None:
    """Copy files matched with a glob pattern in a directory tree to another."""
    for src in src_dir.glob(pattern):
        dest = dest_dir / src.relative_to(src_dir)
        if src.is_dir():
            # NOTE: inefficient if subdirectories also match to the pattern.
            copy_files(src, dest, "*")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            distutils_log.info(f"copied {src} to {dest}")


if __name__ == "__main__":
    build({})
