#!/usr/bin/env python3
"""
scripts/05_sync_to_sheets.py
────────────────────────────────────────────────────────────────────────────────
Sync experiment results to the Google Sheet tracker.

Reads credentials from .env:
  GSHEETS_SA_JSON          — service account JSON (string or path to JSON file)
  GSHEETS_SPREADSHEET_ID   — spreadsheet ID from the URL

Usage
-----
  # Set up / reset headers only (clears old LaBraM rows):
  python scripts/05_sync_to_sheets.py --reset-headers

  # Add / update a result row from experiments/results/results.json:
  python scripts/05_sync_to_sheets.py --results experiments/results/results.json

  # Both at once (first run):
  python scripts/05_sync_to_sheets.py --reset-headers --results experiments/results/results.json

Sheet layout
------------
  Row 1 : section labels  (IDENTITY / CONFIG / BASELINE / TRAINED / Δ / PER-CLASS F1 / JOB)
  Row 2 : column headers
  Row 3+ : one row per experiment run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


# ─── Column definitions ────────────────────────────────────────────────────────

SECTION_LABELS = [
    # (start_col_0idx, span, label)
    (0,  4, "IDENTITY"),
    (4,  4, "CONFIG"),
    (8,  4, "BASELINE  (untrained model)"),
    (12, 4, "TRAINED  (from scratch on LEAD)"),
    (16, 2, "Δ IMPROVEMENT"),
    (18, 3, "PER-CLASS F1  (trained)"),
    (21, 3, "JOB METADATA"),
]

COLUMN_HEADERS = [
    # IDENTITY
    "Exp #", "Run Name", "W&B URL", "Date",
    # CONFIG
    "Model", "Init", "Dataset", "Status",
    # BASELINE (untrained)
    "Base Bal-Acc", "Base Macro F1", "Base Cohen κ", "Base OvR AUC",
    # TRAINED (from scratch)
    "Train Bal-Acc", "Train Macro F1", "Train Cohen κ", "Train OvR AUC",
    # Δ IMPROVEMENT
    "Δ Bal-Acc", "Δ Macro F1",
    # PER-CLASS F1
    "F1 AD", "F1 FTD", "F1 CN",
    # JOB METADATA
    "KOA Job ID", "GPU Hours", "Notes",
]

N_COLS = len(COLUMN_HEADERS)  # 24


# ─── Google Sheets helpers ─────────────────────────────────────────────────────

def _col_letter(idx: int) -> str:
    """0-based column index → letter (A, B, ..., Z, AA, ...)"""
    result = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        result = chr(65 + rem) + result
    return result


def connect_sheet(spreadsheet_id: str, sa_json: str):
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # sa_json can be a JSON string or a path to a file
    if sa_json.strip().startswith("{"):
        info = json.loads(sa_json)
    else:
        info = json.loads(Path(sa_json).read_text())

    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(spreadsheet_id).sheet1


def write_headers(sheet) -> None:
    """Write section labels (row 1) and column headers (row 2)."""
    # Build section label row
    section_row = [""] * N_COLS
    for start, span, label in SECTION_LABELS:
        section_row[start] = label

    sheet.update("A1", [section_row, COLUMN_HEADERS], value_input_option="USER_ENTERED")

    # Apply bold to header rows
    sheet.format("A1:X2", {"textFormat": {"bold": True}})

    # Freeze first two rows
    sheet.spreadsheet.batch_update({
        "requests": [{
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet.id,
                    "gridProperties": {"frozenRowCount": 2},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        }]
    })
    print("Headers written (rows 1–2).")


def clear_data_rows(sheet) -> None:
    """Clear all data rows below the headers."""
    last_row = max(sheet.row_count, 10)
    if last_row > 2:
        sheet.batch_clear([f"A3:{_col_letter(N_COLS - 1)}{last_row}"])
    print("Old data rows cleared.")


def next_exp_number(sheet) -> int:
    """Return the next experiment number (1-indexed)."""
    values = sheet.col_values(1)  # column A
    data_values = [v for v in values[2:] if v.strip()]  # skip header rows
    return len(data_values) + 1


def build_result_row(exp_num: int, results: dict, run_name: str = "") -> list:
    """Build a data row from a results.json dict."""
    baseline = results.get("baseline", {})
    trained  = results.get("trained",  {})

    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and v != v):  # nan check
            return "—"
        return f"{float(v):.4f}"

    def delta(key):
        b = baseline.get(key)
        t = trained.get(key)
        if b is None or t is None:
            return "—"
        d = float(t) - float(b)
        return f"{d:+.4f}"

    return [
        exp_num,                                             # Exp #
        run_name or f"run-{exp_num:02d}",                   # Run Name
        "—",                                                 # W&B URL
        datetime.now().strftime("%Y-%m-%d"),                 # Date
        "EEGConformer",                                      # Model
        "scratch",                                           # Init
        "ADFTD-RS L400",                                     # Dataset
        "complete",                                          # Status
        fmt(baseline.get("balanced_accuracy")),              # Base Bal-Acc
        fmt(baseline.get("f1_macro")),                       # Base Macro F1
        fmt(baseline.get("cohen_kappa")),                    # Base Cohen κ
        fmt(baseline.get("roc_auc_ovr")),                    # Base OvR AUC
        fmt(trained.get("balanced_accuracy")),               # Train Bal-Acc
        fmt(trained.get("f1_macro")),                        # Train Macro F1
        fmt(trained.get("cohen_kappa")),                     # Train Cohen κ
        fmt(trained.get("roc_auc_ovr")),                     # Train OvR AUC
        delta("balanced_accuracy"),                          # Δ Bal-Acc
        delta("f1_macro"),                                   # Δ Macro F1
        fmt(trained.get("acc_AD")),                          # F1 AD
        fmt(trained.get("acc_FTD")),                         # F1 FTD
        fmt(trained.get("acc_CN")),                          # F1 CN
        "—",                                                 # KOA Job ID
        "—",                                                 # GPU Hours
        "",                                                  # Notes
    ]


def append_result_row(sheet, row: list) -> None:
    """Append a result row after the last data row."""
    exp_num = row[0]
    # Check if a row with this exp_num already exists and overwrite it
    all_values = sheet.get_all_values()
    for i, existing_row in enumerate(all_values[2:], start=3):  # skip 2 header rows
        if existing_row and str(existing_row[0]) == str(exp_num):
            sheet.update(f"A{i}", [row], value_input_option="USER_ENTERED")
            print(f"Updated existing row {i} (Exp #{exp_num}).")
            return
    # Otherwise append
    sheet.append_row(row, value_input_option="USER_ENTERED")
    print(f"Appended new row for Exp #{exp_num}.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync experiment results to Google Sheets.")
    parser.add_argument("--reset-headers", action="store_true",
                        help="Rewrite section labels and column headers (clears old rows too).")
    parser.add_argument("--results", default=None,
                        help="Path to results.json to add/update a data row.")
    parser.add_argument("--run-name", default="",
                        help="Override the run name in the sheet row.")
    parser.add_argument("--exp-num", type=int, default=None,
                        help="Explicit experiment number (auto-increments if omitted).")
    parser.add_argument("--env", default=".env",
                        help="Path to .env file (default: .env).")
    args = parser.parse_args()

    load_dotenv(args.env)
    sa_json = os.getenv("GSHEETS_SA_JSON", "")
    spreadsheet_id = os.getenv("GSHEETS_SPREADSHEET_ID", "")

    if not sa_json or not spreadsheet_id:
        raise SystemExit(
            "Missing GSHEETS_SA_JSON or GSHEETS_SPREADSHEET_ID in .env. "
            "Run `koa auth sync` to push credentials to the cluster, or set them locally."
        )

    print("Connecting to Google Sheets...")
    sheet = connect_sheet(spreadsheet_id, sa_json)
    print(f"Connected: {sheet.spreadsheet.title} / {sheet.title}")

    if args.reset_headers:
        clear_data_rows(sheet)
        write_headers(sheet)

    if args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            raise SystemExit(f"Results file not found: {results_path}")
        results = json.loads(results_path.read_text())
        exp_num = args.exp_num or next_exp_number(sheet)
        row = build_result_row(exp_num, results, run_name=args.run_name)
        append_result_row(sheet, row)

    if not args.reset_headers and not args.results:
        parser.print_help()


if __name__ == "__main__":
    main()
