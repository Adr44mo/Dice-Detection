#!/usr/bin/env python3
"""Download the Roboflow dice dataset just like in the notebooks.

Edit the configuration block below (workspace/project/version/output) or
pass a `ROBOFLOW_API_KEY` environment variable before running:

    export ROBOFLOW_API_KEY="your_key"
    python scripts/download_roboflow_dataset.py
"""

import os
from pathlib import Path
from typing import Optional

from roboflow import Roboflow

# ---------------------------------------------------------------------------
# Configuration (adjust as needed)
# ---------------------------------------------------------------------------
ROBOFLOW_WORKSPACE = "workspace-spezm"
ROBOFLOW_PROJECT = "dice-0sexk"
ROBOFLOW_VERSION = 2
DOWNLOAD_FORMAT = "coco"  # e.g. "coco", "yolov8", etc.
# Folder where the Roboflow download will be stored (defaults to ./dice-2)
OUTPUT_DIR = Path("dice-2")
API_KEY = os.environ.get("ROBOFLOW_API_KEY")


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    data_format: str,
    location: Optional[Path] = None,
) -> Path:
    """Download the dataset from Roboflow and return its location."""
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    version_obj = rf_project.version(version)
    dataset = version_obj.download(
        data_format,
        location=str(location) if location else None
    )
    # Roboflow returns an object with `.location` describing the dataset root
    return Path(dataset.location)


def main() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY environment variable not set. "
            "Export it before running this script."
        )

    target_dir = OUTPUT_DIR.resolve()
    target_dir_parent = target_dir.parent
    target_dir_parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        if any(target_dir.iterdir()):
            raise FileExistsError(
                f"Target directory {target_dir} already exists and is not empty. "
                "Delete it or change OUTPUT_DIR before downloading again."
            )
        target_dir.rmdir()

    dataset_path = download_dataset(
        api_key=API_KEY,
        workspace=ROBOFLOW_WORKSPACE,
        project=ROBOFLOW_PROJECT,
        version=ROBOFLOW_VERSION,
        data_format=DOWNLOAD_FORMAT,
        location=target_dir,
    )
    dataset_path = dataset_path.resolve()

    try:
        relative = dataset_path.relative_to(Path.cwd())
    except ValueError:
        relative = dataset_path

    print(f"Dataset downloaded to: {relative}")


if __name__ == "__main__":
    main()
