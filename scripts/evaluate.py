from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspicio.config import load_config
from inspicio.evaluation import EvaluationStats, evaluate_retrieval
from inspicio.utils import parse_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Inspicio JSONL file.")
    parser.add_argument("--config", required=True, help="YAML or JSON config file.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--report-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stats = EvaluationStats(cfg.retrieval.top_k)
    with open(args.input_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            retrieved = obj.get("RETRIEVED_SYNSETS", [])
            gt = obj.get("GROUND_TRUTH")
            if isinstance(gt, dict):
                gold_ids = [str(x) for x in gt.get("id", [])]
            else:
                gold_ids, _ = parse_ground_truth(obj.get(cfg.columns.ground_truth, ""))
            stats.add_result(evaluate_retrieval(retrieved, gold_ids))
    report = stats.report("INSPICIO WSD EVALUATION REPORT", [f"  Input JSONL: {args.input_jsonl}"])
    report_file = args.report_file or cfg.report_file
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
