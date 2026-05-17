from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspicio.config import load_config, merge_cli_overrides
from inspicio.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Inspicio WSD pipeline.")
    parser.add_argument("--config", required=True, help="YAML or JSON config file.")
    parser.add_argument("--mode", choices=["full", "retrieval"], help="Override config mode.")
    parser.add_argument("--input-csv", help="Override input CSV path.")
    parser.add_argument("--input-jsonl", help="Override retrieval-mode JSONL input.")
    parser.add_argument("--output-jsonl", help="Override output JSONL path.")
    parser.add_argument("--report-file", help="Override report path.")
    parser.add_argument("--no-resume", action="store_true", help="Process all rows even if output exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    merge_cli_overrides(
        cfg,
        mode=args.mode,
        input_csv=args.input_csv,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        report_file=args.report_file,
    )
    if args.no_resume:
        cfg.resume = False
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
