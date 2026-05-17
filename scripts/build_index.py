from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspicio.config import load_config
from inspicio.indexing import build_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OEWN Chroma indexes for one or more PoS.")
    parser.add_argument("--config", required=True, help="YAML or JSON config file.")
    parser.add_argument("--pos", choices=["all", "verb", "noun", "adjective", "adverb"], help="Override index.pos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.pos:
        cfg.index.pos = args.pos
    build_indexes(cfg.index, cfg.embedder)


if __name__ == "__main__":
    main()
