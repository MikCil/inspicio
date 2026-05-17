from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .config import RunConfig, ensure_parent, row_pos
from .embedders import create_embedder
from .evaluation import EvaluationStats, evaluate_retrieval
from .llm import ChatLLM
from .retriever import RetrieverRegistry, retrieve_synsets
from .utils import parse_ground_truth


def load_processed_ids_jsonl(output_path: str, id_col: str) -> Set[str]:
    processed = set()
    if not Path(output_path).exists():
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if id_col in obj:
                    processed.add(str(obj[id_col]))
    except Exception as exc:
        print(f"  Warning: could not read resume file: {exc}")
    return processed


def _jsonish(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _config_lines(cfg: RunConfig) -> List[str]:
    return [
        f"  Mode:                    {cfg.mode}",
        f"  Input CSV:               {cfg.input_csv}",
        f"  Input JSONL:             {cfg.input_jsonl or ''}",
        f"  Output JSONL:            {cfg.output_jsonl}",
        f"  Embedding model:         {cfg.embedder.model}",
        f"  LLM provider/model:      {cfg.llm.provider} / {cfg.llm.model}",
        f"  Translation enabled:     {cfg.generation.translation_enabled}",
        f"  Top-K:                   {cfg.retrieval.top_k}",
        f"  Dense top-N/definition:  {cfg.retrieval.dense_top_n_per_definition}",
        f"  MMR enabled:             {cfg.retrieval.enable_mmr}",
        f"  Lexname coverage:        {cfg.retrieval.enable_lexname_coverage}",
    ]


class WSDPipeline:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.embedder = create_embedder(cfg.embedder)
        self.retrievers = RetrieverRegistry(cfg.index, cfg.retrieval)
        self.llm: Optional[ChatLLM] = None
        if cfg.mode == "full":
            self.llm = ChatLLM(cfg.llm)

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        columns = cfg.columns
        pos = row_pos(row, columns, cfg.default_pos)
        language = (row.get(columns.language) or "").strip() or "the source language"
        token = row.get(columns.token, "") or ""
        lemma = row.get(columns.lemma, "") or ""
        sentence = row.get(columns.sentence, "") or ""
        ground_truth_ids, ground_truth_glosses = parse_ground_truth(row.get(columns.ground_truth, ""))

        if self.llm is None:
            raise RuntimeError("Full mode requires an LLM client.")

        if cfg.generation.translation_enabled:
            translations = self.llm.translate_sentence(
                language,
                sentence,
                temperature=cfg.generation.translation_temperature,
            )
            time.sleep(cfg.llm.request_delay)
        else:
            translations = {}

        enriched = self.llm.generate_enriched_gloss(
            language=language,
            token=token,
            lemma=lemma,
            sentence=sentence,
            translations=translations,
            pos=pos,
            temperature=cfg.generation.gloss_temperature,
            max_definitions=cfg.generation.max_definitions,
            max_lemmas=cfg.generation.max_lemmas,
        )
        time.sleep(cfg.llm.request_delay)
        return self._retrieve(row, pos, translations, enriched, ground_truth_ids, ground_truth_glosses)

    def process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        columns = cfg.columns
        pos = row_pos(entry, columns, cfg.default_pos)
        if isinstance(entry.get("GROUND_TRUTH"), dict):
            ground_truth_ids = [str(x) for x in entry["GROUND_TRUTH"].get("id", [])]
            ground_truth_glosses = [str(x) for x in entry["GROUND_TRUTH"].get("gloss", [])]
        else:
            ground_truth_ids, ground_truth_glosses = parse_ground_truth(entry.get(columns.ground_truth, ""))

        translations = _jsonish(entry.get("GENERATED_TRANSLATIONS"), {})
        enriched = _jsonish(entry.get("GENERATED_ENRICHED"), {})
        if not enriched:
            raise ValueError("Retrieval mode requires GENERATED_ENRICHED in each JSONL entry.")
        return self._retrieve(entry, pos, translations, enriched, ground_truth_ids, ground_truth_glosses)

    def _retrieve(
        self,
        row: Dict[str, Any],
        pos: str,
        translations: Dict[str, str],
        enriched: Dict[str, Any],
        ground_truth_ids: List[str],
        ground_truth_glosses: List[str],
    ) -> Dict[str, Any]:
        definitions = [str(x).strip() for x in enriched.get("definitions", []) if str(x).strip()]
        candidate_lemmas = [str(x).strip() for x in enriched.get("candidate_lemmas", []) if str(x).strip()]
        retriever = self.retrievers.get(pos)
        retrieval = retrieve_synsets(definitions, candidate_lemmas, self.embedder, retriever, self.cfg.retrieval)
        retrieved = retrieval["retrieved_synsets"]
        evaluation = evaluate_retrieval(retrieved, ground_truth_ids)
        retrieval_meta = retrieval["retrieval_meta"]
        retrieval_meta.update(
            {
                "pos": pos,
                "language": (row.get(self.cfg.columns.language) or "").strip(),
                "translation_enabled": self.cfg.generation.translation_enabled,
            }
        )
        return {
            "translations": translations,
            "enriched": {
                "definitions": retrieval["definitions"],
                "candidate_lemmas": candidate_lemmas,
            },
            "ground_truth": {"id": ground_truth_ids, "gloss": ground_truth_glosses},
            "evaluation": evaluation,
            "retrieval_meta": retrieval_meta,
            "retrieved_synsets": retrieved,
        }


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_pipeline(cfg: RunConfig) -> EvaluationStats:
    ensure_parent(cfg.output_jsonl)
    ensure_parent(cfg.report_file)
    pipeline = WSDPipeline(cfg)
    stats = EvaluationStats(cfg.retrieval.top_k)

    if cfg.mode == "retrieval":
        if not cfg.input_jsonl:
            raise ValueError("Retrieval mode requires input_jsonl.")
        rows = list(_iter_jsonl(cfg.input_jsonl))
    elif cfg.mode == "full":
        rows = _load_csv(cfg.input_csv)
    else:
        raise ValueError("mode must be 'full' or 'retrieval'.")

    processed_ids = load_processed_ids_jsonl(cfg.output_jsonl, cfg.columns.id) if cfg.resume else set()
    print(f"Loaded {len(rows)} rows")
    if processed_ids:
        print(f"Skipping {len(processed_ids)} already processed rows")

    with open(cfg.output_jsonl, "a", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            row_id = str(row.get(cfg.columns.id, idx))
            if row_id in processed_ids:
                stats.add_skipped()
                continue
            token = str(row.get(cfg.columns.token, ""))[:40]
            print(f"[{idx}/{len(rows)}] ID={row_id} token={token}")
            try:
                result = pipeline.process_entry(row) if cfg.mode == "retrieval" else pipeline.process_row(row)
                output = dict(row)
                output.update(
                    {
                        cfg.columns.pos: result["retrieval_meta"]["pos"],
                        "GENERATED_TRANSLATIONS": result["translations"],
                        "GENERATED_ENRICHED": result["enriched"],
                        "GROUND_TRUTH": result["ground_truth"],
                        "EVALUATION": result["evaluation"],
                        "RETRIEVAL_META": result["retrieval_meta"],
                        "RETRIEVED_SYNSETS": result["retrieved_synsets"],
                    }
                )
                out.write(json.dumps(output, ensure_ascii=False) + "\n")
                out.flush()
                stats.add_result(result["evaluation"])
                rank = result["evaluation"].get("rank")
                print(f"  rank={rank if rank is not None else 'n/a'} gt={result['evaluation'].get('has_ground_truth')}")
            except Exception as exc:
                stats.add_error()
                output = dict(row)
                output["ERROR"] = str(exc)
                out.write(json.dumps(output, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  Error: {exc}")

    report = stats.report("INSPICIO WSD EVALUATION REPORT", _config_lines(cfg))
    Path(cfg.report_file).write_text(report, encoding="utf-8")
    print(report)
    return stats
