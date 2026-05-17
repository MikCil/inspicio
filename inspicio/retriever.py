from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import RetrievalConfig
from .utils import cosine_sim, l2_normalize, normalize_synset_id, split_lemmas


class LemmaIndex:
    def __init__(self, collection, cache_path: str, batch_size: int):
        self.collection = collection
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.index: Dict[str, Set[str]] = {}

    def load_or_build(self) -> None:
        if Path(self.cache_path).exists():
            try:
                raw = json.loads(Path(self.cache_path).read_text(encoding="utf-8"))
                self.index = {k: set(v) for k, v in raw.items()}
                print(f"  Loaded lemma index: {self.cache_path} ({len(self.index):,} lemmas)")
                return
            except Exception as exc:
                print(f"  Warning: failed to load lemma cache, rebuilding: {exc}")

        print("  Building lemma index from Chroma metadata...")
        idx: Dict[str, Set[str]] = defaultdict(set)
        offset = 0
        total = self.collection.count()
        while offset < total:
            batch = self.collection.get(
                include=["metadatas"],
                limit=min(self.batch_size, total - offset),
                offset=offset,
            )
            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            for syn_id, meta in zip(ids, metas):
                for lemma in split_lemmas((meta or {}).get("lemmas_str", "")):
                    idx[lemma].add(syn_id)
            if not ids:
                break
            offset += len(ids)

        self.index = dict(idx)
        try:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cache_path).write_text(
                json.dumps({k: sorted(v) for k, v in self.index.items()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"  Warning: could not save lemma cache: {exc}")

    def lookup(self, lemma: str) -> Set[str]:
        key = " ".join((lemma or "").strip().lower().split())
        return set(self.index.get(key, set()))


class SynsetRetriever:
    def __init__(self, chroma_path: str, collection_name: str, lemma_cache: str, config: RetrievalConfig):
        self.config = config
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to load retrieval indexes.") from exc
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        self.doc_count = self.collection.count()
        print(f"  Loaded collection '{collection_name}' with {self.doc_count:,} documents")
        self.lemma_index = LemmaIndex(self.collection, lemma_cache, config.lemma_index_batch_size)
        self.lemma_index.load_or_build()

    def dense_query(self, definition_embedding: List[float], n_results: int) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[definition_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        ids = result["ids"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        docs = result["documents"][0] if result.get("documents") else ["" for _ in ids]
        embeddings = result["embeddings"][0] if result.get("embeddings") else [None for _ in ids]

        out = []
        for rank, (sid, meta, distance, doc, embedding) in enumerate(
            zip(ids, metas, distances, docs, embeddings), start=1
        ):
            out.append(
                {
                    "synset_id": sid,
                    "normalized_id": normalize_synset_id(sid),
                    "lemmas": (meta or {}).get("lemmas_str", ""),
                    "gloss": (meta or {}).get("gloss", ""),
                    "lexname": (meta or {}).get("lexname", ""),
                    "distance": float(distance),
                    "document": doc or "",
                    "embedding": embedding,
                    "rank_in_definition": rank,
                }
            )
        return out

    def get_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        result = self.collection.get(ids=ids, include=["metadatas", "documents", "embeddings"])
        out = {}
        for sid, meta, doc, embedding in zip(
            result["ids"], result["metadatas"], result["documents"], result["embeddings"]
        ):
            out[sid] = {
                "synset_id": sid,
                "normalized_id": normalize_synset_id(sid),
                "lemmas": (meta or {}).get("lemmas_str", ""),
                "gloss": (meta or {}).get("gloss", ""),
                "lexname": (meta or {}).get("lexname", ""),
                "document": doc or "",
                "embedding": embedding,
            }
        return out

    def lemma_pool_ids(self, candidate_lemmas: List[str]) -> Set[str]:
        ids: Set[str] = set()
        for lemma in candidate_lemmas:
            ids |= self.lemma_index.lookup(lemma)
        return ids


class RetrieverRegistry:
    def __init__(self, index_config, retrieval_config: RetrievalConfig):
        self.index_config = index_config
        self.retrieval_config = retrieval_config
        self._cache: Dict[str, SynsetRetriever] = {}

    def get(self, pos: str) -> SynsetRetriever:
        if pos not in self._cache:
            loc = self.index_config.location_for_pos(pos)
            collection = loc["collection"]
            lemma_cache = loc.get("lemma_cache") or self.index_config.lemma_cache_for(collection, pos)
            self._cache[pos] = SynsetRetriever(
                loc["chroma_path"],
                collection,
                lemma_cache,
                self.retrieval_config,
            )
        return self._cache[pos]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def get_lexname_representatives(candidates: List[Dict[str, Any]], min_per_lexname: int) -> Set[int]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        lexname = (candidate.get("lexname") or "").strip()
        if lexname:
            grouped[lexname].append(idx)
    reps: Set[int] = set()
    for indices in grouped.values():
        reps.update(indices[:min_per_lexname])
    return reps


def mmr_rerank_with_scores(
    candidate_embeddings: np.ndarray,
    relevance_scores: np.ndarray,
    k: int,
    lambda_param: float,
    preselected_indices: Optional[Set[int]] = None,
) -> List[int]:
    preselected_indices = preselected_indices or set()
    n = candidate_embeddings.shape[0]
    if n <= k:
        return list(range(n))

    sim_matrix = compute_similarity_matrix(candidate_embeddings)
    selected = list(preselected_indices)
    remaining = set(range(n)) - preselected_indices

    if not selected and remaining:
        best = max(remaining, key=lambda i: relevance_scores[i])
        selected.append(best)
        remaining.remove(best)

    while len(selected) < k and remaining:
        best_score = float("-inf")
        best_idx = None
        for idx in remaining:
            max_sim = max(sim_matrix[idx, selected_idx] for selected_idx in selected) if selected else 0.0
            score = lambda_param * relevance_scores[idx] - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def retrieve_synsets(
    definitions: List[str],
    candidate_lemmas: List[str],
    embedder,
    retriever: SynsetRetriever,
    config: RetrievalConfig,
) -> Dict[str, Any]:
    unique_definitions = []
    seen = set()
    for definition in definitions:
        key = definition.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_definitions.append(definition)

    definition_embeddings = [
        l2_normalize(np.array(embedder.embed(definition, kind="query"), dtype=np.float32))
        for definition in unique_definitions
    ]
    weights = list(config.definition_decay_weights[: len(definition_embeddings)])
    if len(weights) < len(definition_embeddings):
        next_weight = weights[-1] if weights else 1.0
        while len(weights) < len(definition_embeddings):
            next_weight *= 0.5
            weights.append(next_weight)

    merged: Dict[str, Dict[str, Any]] = {}
    dense_meta = {
        "definitions_used": unique_definitions,
        "weights_used": weights,
        "dense_top_n_per_definition": config.dense_top_n_per_definition,
        "dense_queries": [],
    }

    for def_idx, (definition, definition_embedding, weight) in enumerate(
        zip(unique_definitions, definition_embeddings, weights), start=1
    ):
        dense_results = retriever.dense_query(definition_embedding.tolist(), config.dense_top_n_per_definition)
        dense_meta["dense_queries"].append(
            {
                "definition_index": def_idx,
                "definition": definition,
                "weight": weight,
                "returned": len(dense_results),
            }
        )
        for item in dense_results:
            syn_embedding = item.get("embedding")
            if syn_embedding is None:
                continue
            syn_embedding_np = l2_normalize(np.array(syn_embedding, dtype=np.float32))
            sim = cosine_sim(definition_embedding, syn_embedding_np)
            contribution = weight * sim
            normalized_id = item["normalized_id"]
            rec = merged.get(normalized_id)
            if rec is None:
                rec = {
                    "synset_id": item["synset_id"],
                    "normalized_id": normalized_id,
                    "lemmas": item.get("lemmas", ""),
                    "gloss": item.get("gloss", ""),
                    "lexname": item.get("lexname", ""),
                    "embedding": syn_embedding_np,
                    "distance_debug": item.get("distance"),
                    "sources_dense": [],
                    "lemma_matched": False,
                    "s_base": 0.0,
                    "s_final": 0.0,
                }
                merged[normalized_id] = rec
            rec["s_base"] += contribution
            rec["sources_dense"].append(
                {
                    "definition_index": def_idx,
                    "rank_in_definition": item.get("rank_in_definition"),
                    "similarity": sim,
                    "weight": weight,
                    "contribution": contribution,
                }
            )

    lemma_ids = retriever.lemma_pool_ids(candidate_lemmas)
    lemma_records = retriever.get_by_ids(sorted(lemma_ids))
    for syn_id, synrec in lemma_records.items():
        normalized_id = synrec["normalized_id"]
        rec = merged.get(normalized_id)
        if rec is None:
            embedding = synrec.get("embedding")
            rec = {
                "synset_id": syn_id,
                "normalized_id": normalized_id,
                "lemmas": synrec.get("lemmas", ""),
                "gloss": synrec.get("gloss", ""),
                "lexname": synrec.get("lexname", ""),
                "embedding": l2_normalize(np.array(embedding, dtype=np.float32)) if embedding is not None else None,
                "distance_debug": None,
                "sources_dense": [],
                "lemma_matched": True,
                "s_base": 0.0,
                "s_final": 0.0,
            }
            merged[normalized_id] = rec
        else:
            rec["lemma_matched"] = True

    for rec in merged.values():
        if rec["s_base"] > 0.0:
            boost = config.lemma_multiplier_gamma if rec["lemma_matched"] else 0.0
            rec["s_final"] = rec["s_base"] * (1.0 + boost)
        elif rec["lemma_matched"]:
            rec["s_final"] = float(config.lemma_only_default_score)

    if config.normalize_final_scores and merged:
        max_score = max(rec["s_final"] for rec in merged.values())
        if max_score > 1e-12:
            for rec in merged.values():
                rec["s_final"] /= max_score

    pool = sorted(merged.values(), key=lambda item: item["s_final"], reverse=True)
    if len(pool) > config.max_candidates_before_mmr:
        pool = pool[: config.max_candidates_before_mmr]
    pool = [item for item in pool if item.get("embedding") is not None]

    if config.enable_mmr and pool:
        preselected = (
            get_lexname_representatives(pool, config.lexname_min_candidates)
            if config.enable_lexname_coverage
            else set()
        )
        selected = mmr_rerank_with_scores(
            np.stack([item["embedding"] for item in pool], axis=0),
            np.array([item["s_final"] for item in pool], dtype=np.float32),
            min(config.top_k, len(pool)),
            config.mmr_lambda,
            preselected,
        )
        final = [pool[idx] for idx in selected][: config.top_k]
    else:
        final = pool[: config.top_k]

    retrieved = []
    for rank, item in enumerate(final, start=1):
        retrieved.append(
            {
                "rank": rank,
                "synset_id": item["synset_id"],
                "normalized_id": item["normalized_id"],
                "lemmas": item["lemmas"],
                "gloss": item["gloss"],
                "lexname": item["lexname"],
                "lemma_matched": item["lemma_matched"],
                "s_base": round(float(item["s_base"]), 6),
                "s_final": round(float(item["s_final"]), 6),
                "sources_dense": item["sources_dense"],
            }
        )

    return {
        "retrieved_synsets": retrieved,
        "retrieval_meta": {
            "candidate_lemmas": candidate_lemmas,
            "lemma_pool_size": len(lemma_records),
            "merged_pool_size_before_cap": len(merged),
            "merged_pool_size_for_mmr": len(pool),
            "mmr_enabled": config.enable_mmr,
            "mmr_lambda": config.mmr_lambda if config.enable_mmr else None,
            "lexname_coverage_enabled": config.enable_lexname_coverage,
            "dense_meta": dense_meta,
        },
        "definitions": unique_definitions,
    }
