#!/usr/bin/env python3
"""
scripts/01_download_lead.py
────────────────────────────────────────────────────────────────────────────────
Download the LEAD dataset from Google Drive.

LEAD is hosted on Google Drive. This script uses gdown to fetch the files.
You can also download them manually from the link below.

  GitHub : https://github.com/DL4mHealth/LEAD
  Drive  : https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX

Usage
-----
  python scripts/01_download_lead.py --output-dir data/lead/source

  # Download only the ADFTD-RS L400 subset (recommended for training):
  python scripts/01_download_lead.py --output-dir data/lead/source --subset ADFTD-RS

Directory layout after download
--------------------------------
  data/lead/source/
  └── ADFTD-RS/              ← resting-state AD / FTD / CN windows (L400)
      ├── X.dat
      ├── y.dat
      └── meta.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LEAD_FOLDER_URL = "https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX"

# Known Google Drive IDs for individual subsets (update if LEAD repo changes them)
SUBSET_IDS: dict[str, str] = {
    # Map subset name → folder ID (add more as needed)
    # These IDs are taken from the LEAD GitHub README.
    # If a download fails, visit the Drive link above and grab the folder ID manually.
}


def _check_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is required: pip install gdown")
        sys.exit(1)


def download_folder(folder_url: str, output_dir: Path) -> None:
    import gdown

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LEAD folder → {output_dir}")
    gdown.download_folder(url=folder_url, output=str(output_dir), quiet=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LEAD dataset from Google Drive.")
    parser.add_argument(
        "--output-dir", default="data/lead/source",
        help="Local directory to download into (default: data/lead/source)",
    )
    parser.add_argument(
        "--subset", default=None,
        help=(
            "Specific LEAD subset to download, e.g. ADFTD-RS. "
            "If omitted, prints the Drive link for manual download."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.subset and args.subset in SUBSET_IDS:
        _check_gdown()
        folder_id = SUBSET_IDS[args.subset]
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        download_folder(url, output_dir / args.subset)
    else:
        # Fall back to full folder download or print instructions
        _check_gdown()
        print(
            "\nDownloading the full LEAD dataset folder.\n"
            "This may take a while depending on your connection.\n"
            f"You can also download manually from:\n  {LEAD_FOLDER_URL}\n"
        )
        download_folder(LEAD_FOLDER_URL, output_dir)

    print(f"\nDone. LEAD data is at: {output_dir.resolve()}")
    print("Next step: python scripts/02_import_lead.py --source-root", output_dir)


if __name__ == "__main__":
    main()
