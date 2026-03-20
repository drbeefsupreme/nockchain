from __future__ import annotations

import shutil
from pathlib import Path


def copy_directory_contents(source_dir: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.iterdir():
        destination = target_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination)
    return target_dir
