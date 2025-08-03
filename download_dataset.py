"""Utilities for downloading datasets from the UCI repository.

This module provides small helper functions that download and unzip
public datasets used throughout the project. The functions are intentionally
lightweight so they can be imported and reused in tests or scripts without
side effects.
"""

from __future__ import annotations

import os
from urllib import request
import zipfile

from typing import Dict


# Base directory where datasets will be stored
DATA_DIR = "data"


# Mapping from dataset name to the corresponding UCI download URL
NAME_URL_DICT_UCI: Dict[str, str] = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
}


def unzip_file(zip_filepath: str, dest_path: str) -> None:
    """Extract a ZIP archive to a destination directory.

    Args:
        zip_filepath: Path to the ZIP file to be extracted.
        dest_path: Directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        # Extract all files into the destination directory
        zip_ref.extractall(dest_path)


def download_from_uci(name: str) -> None:
    """Download and extract a dataset from the UCI repository.

    The dataset is downloaded only if it does not already exist on disk.

    Args:
        name: Name of the dataset to download. Must exist in
            :data:`NAME_URL_DICT_UCI`.
    """

    print(f"Start processing dataset {name} from UCI.")
    save_dir = f"{DATA_DIR}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f"{save_dir}/{name}.zip")
        print(
            f"Finish downloading dataset from {url}, data has been saved to {save_dir}."
        )

        unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")

    else:
        print("Aready downloaded.")


if __name__ == "__main__":
    for dataset_name in NAME_URL_DICT_UCI.keys():
        download_from_uci(dataset_name)

