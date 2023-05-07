"""Build script."""

import shutil
from distutils import log as distutils_log
from pathlib import Path
from typing import Any, Dict
from whisper_cpp_file_gen import WhisperCppFileGen

import skbuild
import skbuild.constants

__all__ = ("build",)


def build(setup_kwargs: Dict[str, Any]) -> None:
    """Build C-extensions."""
    skbuild.setup(**setup_kwargs, script_args=["build_ext"])

    src_dir = Path(skbuild.constants.CMAKE_INSTALL_DIR()) / "lib"
    dest_dir = Path("whisper_cpp_python")

    # Delete C-extensions copied in previous runs, just in case.
    remove_files(dest_dir, "**/*.so")
    remove_files(dest_dir, "**/*.dll")
    remove_files(dest_dir, "**/*.dylib")

    # Copy built C-extensions back to the project.
    copy_files(src_dir, dest_dir, "**/*.so")
    copy_files(src_dir, dest_dir, "**/*.dll")
    copy_files(src_dir, dest_dir, "**/*.dylib")

    # generate whisper_cpp.py with whisper.h header file
    c_header_file = Path(skbuild.constants.CMAKE_INSTALL_DIR()) / "include" / "whisper.h"
    file_gen = WhisperCppFileGen(c_header_file)
    file_gen.output(dest_dir / "whisper_cpp.py")


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
    build({'packages': ['whisper_cpp_python']})
