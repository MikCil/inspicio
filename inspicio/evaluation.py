from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import normalize_synset_id


def evaluate_retrieval(retrieved: List[Dict[str, Any]], ground_truth_ids: List[str]) -> Dict[str, Any]:
    gold_set = {normalize_synset_id(x) for x in (ground_truth_ids or []) if x}
    if not gold_set:
        return {
            "has_ground_truth": False,
            "correct_retrieved": None,
            "rank": None,
            "top_k": len(retrieved),
            "matched_gold_ids": [],
        }

    matched = []
    best_rank = None
    for rank, item in enumerate(retrieved, start=1):
        nid = item.get("normalized_id")
        if nid in gold_set:
            matched.append(nid)
            if best_rank is None:
                best_rank = rank

    return {
        "has_ground_truth": True,
        "correct_retrieved": best_rank is not None,
        "rank": best_rank,
        "top_k": len(retrieved),
        "matched_gold_ids": matched,
    }


@dataclass
class EvaluationStats:
    k: int
    tagged: int = 0
    evaluated: int = 0
    correct_at_k: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    ranks: List[int] = field(default_factory=list)
    errors: int = 0
    skipped: int = 0
    no_ground_truth: int = 0

    def add_result(self, evaluation: Dict[str, Any]) -> None:
        self.tagged += 1
        if not evaluation.get("has_ground_truth"):
            self.no_ground_truth += 1
            return
        self.evaluated += 1
        rank = evaluation.get("rank")
        if rank is not None:
            self.ranks.append(int(rank))
            for n in range(int(rank), self.k + 1):
                self.correct_at_k[n] += 1

    def add_error(self) -> None:
        self.errors += 1

    def add_skipped(self) -> None:
        self.skipped += 1

    def recall_at(self, k: int) -> float:
        return 0.0 if self.evaluated == 0 else self.correct_at_k[k] / self.evaluated

    def mrr(self) -> float:
        return 0.0 if self.evaluated == 0 else sum(1.0 / r for r in self.ranks) / self.evaluated

    def report(self, title: str, config_lines: List[str]) -> str:
        lines = [
            "=" * 70,
            title,
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "CONFIGURATION",
            "-" * 70,
            *config_lines,
            "",
            "-" * 70,
            "DATASET STATISTICS",
            "-" * 70,
            f"  Rows tagged:              {self.tagged}",
            f"  Rows evaluated:           {self.evaluated}",
            f"  Rows without ground truth: {self.no_ground_truth}",
            f"  Rows with errors:         {self.errors}",
            f"  Rows skipped (resume):    {self.skipped}",
            "",
            "-" * 70,
            "RECALL@K",
            "-" * 70,
        ]
        for k in range(1, self.k + 1):
            recall = self.recall_at(k)
            lines.append(f"  Recall@{k:2d}:  {recall:6.2%}")
        lines.extend(["", "-" * 70, "RANKING METRICS", "-" * 70, f"  MRR: {self.mrr():.4f}", "", "=" * 70])
        return "\n".join(lines)
