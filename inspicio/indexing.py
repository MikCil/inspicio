from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Dict, List

from .config import EmbedderConfig, IndexConfig, POS_TO_WN, normalize_pos, pos_values
from .embedders import create_embedder, chunked


def ensure_lexicon(lexicon_id: str) -> None:
    try:
        import wn
    except ImportError as exc:
        raise RuntimeError("Install wn to build OEWN indexes.") from exc
    try:
        wn.download(lexicon_id)
    except wn.Error:
        wn.Wordnet(lexicon_id)


def get_related_synset_info(related_synset) -> Dict[str, Any]:
    return {
        "synset_id": related_synset.id,
        "lemmas": related_synset.lemmas(),
        "gloss": related_synset.definition() or "",
    }


def get_synset_relations(synset, *relation_names: str) -> List[Dict[str, Any]]:
    try:
        return [get_related_synset_info(item) for item in synset.get_related(*relation_names)]
    except Exception:
        return []


def get_sense_level_relations(synset, *relation_names: str) -> List[Dict[str, Any]]:
    related = []
    seen = set()
    for sense in synset.senses():
        try:
            for related_sense in sense.get_related(*relation_names):
                related_synset = related_sense.synset()
                if related_synset.id not in seen:
                    seen.add(related_synset.id)
                    related.append(get_related_synset_info(related_synset))
        except Exception:
            continue
    return related


def get_synset_frames(synset) -> List[str]:
    frames = []
    seen = set()
    for sense in synset.senses():
        try:
            sense_frames = sense.frames()
        except Exception:
            sense_frames = []
        for frame in sense_frames:
            if frame not in seen:
                seen.add(frame)
                frames.append(frame)
    return frames


def get_domain_topics(synset) -> List[str]:
    domains = set()
    for sense in synset.senses():
        for relation in ("has_domain_topic", "domain_topic"):
            try:
                for related_sense in sense.get_related(relation):
                    domains.update(related_sense.synset().lemmas())
            except Exception:
                pass
    return sorted(domains)


def extract_offset_from_id(synset_id: str) -> str:
    for part in synset_id.split("-"):
        if part.isdigit() and len(part) == 8:
            return part
    return ""


def build_synset_document(synset) -> Dict[str, Any]:
    relations = {
        "hypernyms": get_synset_relations(synset, "hypernym", "instance_hypernym"),
        "hyponyms": get_synset_relations(synset, "hyponym", "instance_hyponym"),
        "similar_tos": get_synset_relations(synset, "similar"),
        "verb_groups": get_synset_relations(synset, "verb_group"),
        "entails": get_synset_relations(synset, "entails"),
        "causes": get_synset_relations(synset, "causes"),
        "also_sees": get_synset_relations(synset, "also"),
        "antonyms": get_sense_level_relations(synset, "antonym"),
        "derivationally_related": get_sense_level_relations(synset, "derivation"),
    }
    return {
        "synset_id": synset.id,
        "pos": synset.pos,
        "lemmas": synset.lemmas(),
        "gloss": synset.definition() or "",
        "examples": synset.examples() or [],
        "lexname": synset.lexfile() or "",
        "domains": get_domain_topics(synset),
        "frames": get_synset_frames(synset),
        "relations": relations,
        "meta": {"source": "OEWN", "offset": extract_offset_from_id(synset.id)},
    }


def format_relation_items(items: List[Dict[str, Any]]) -> str:
    return " | ".join(f"{', '.join(item['lemmas'])} - {item['gloss']}" for item in items)


def format_for_embedding(doc: Dict[str, Any]) -> str:
    lines = [f"LEMMA(S): {'; '.join(doc['lemmas'])}", ""]
    if doc["gloss"]:
        lines.extend([f"GLOSS: {doc['gloss']}", ""])
    if doc["frames"]:
        lines.extend([f"FRAMES: {' | '.join(doc['frames'])}", ""])
    if doc["examples"]:
        lines.extend([f"EXAMPLES: {' | '.join(doc['examples'])}", ""])
    for hypernym in doc["relations"]["hypernyms"]:
        lines.append(f"HYPERNYM(S): {', '.join(hypernym['lemmas'])} - {hypernym['gloss']}")
    similar = doc["relations"]["similar_tos"] + doc["relations"]["verb_groups"]
    if similar:
        lines.append(f"SIMILAR/VERB_GROUP: {format_relation_items(similar)}")
    for key, label in (("entails", "ENTAILS"), ("causes", "CAUSES"), ("antonyms", "ANTONYM(S)")):
        if doc["relations"][key]:
            lines.append(f"{label}: {format_relation_items(doc['relations'][key])}")
    domain = []
    if doc["lexname"]:
        domain.append(doc["lexname"])
    domain.extend(doc["domains"])
    if domain:
        lines.append(f"DOMAIN/LEXNAME: {'; '.join(domain)}")
    return "\n".join(lines)


def compute_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    lexnames = Counter(doc["lexname"] for doc in documents if doc["lexname"])
    return {
        "total_synsets": len(documents),
        "total_lemmas": sum(len(doc["lemmas"]) for doc in documents),
        "total_examples": sum(len(doc["examples"]) for doc in documents),
        "synsets_with_examples": sum(1 for doc in documents if doc["examples"]),
        "lexname_distribution": lexnames,
    }


def metadata_for_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "synset_id": doc["synset_id"],
        "pos": doc["pos"],
        "lemmas_json": json.dumps(doc["lemmas"], ensure_ascii=False),
        "lemmas_str": "; ".join(doc["lemmas"]),
        "gloss": doc["gloss"][:500] if doc["gloss"] else "",
        "lexname": doc["lexname"] or "",
        "domains_json": json.dumps(doc["domains"], ensure_ascii=False),
        "num_hypernyms": len(doc["relations"]["hypernyms"]),
        "num_hyponyms": len(doc["relations"]["hyponyms"]),
        "has_examples": len(doc["examples"]) > 0,
        "full_document_json": json.dumps(doc, ensure_ascii=False),
    }


def build_index_for_pos(pos: str, index_config: IndexConfig, embedder_config: EmbedderConfig) -> None:
    try:
        import chromadb
        import wn
    except ImportError as exc:
        raise RuntimeError("Install chromadb and wn to build OEWN indexes.") from exc
    pos_name = normalize_pos(pos)
    wn_pos = POS_TO_WN[pos_name]
    loc = index_config.location_for_pos(pos_name)
    print(f"\nBuilding {pos_name} index -> {loc['chroma_path']} / {loc['collection']}")
    ensure_lexicon(index_config.lexicon_id)
    wordnet = wn.Wordnet(index_config.lexicon_id)
    synsets = wordnet.synsets(pos=wn_pos)
    print(f"  Found {len(synsets):,} synsets")

    documents = [build_synset_document(synset) for synset in synsets]
    embedding_texts = [format_for_embedding(doc) for doc in documents]
    stats = compute_statistics(documents)
    print(
        f"  Lemmas={stats['total_lemmas']:,} examples={stats['total_examples']:,} "
        f"with_examples={stats['synsets_with_examples']:,}"
    )

    embedder = create_embedder(embedder_config)
    chroma_client = chromadb.PersistentClient(path=loc["chroma_path"])
    if index_config.reset_collection:
        try:
            chroma_client.delete_collection(loc["collection"])
        except Exception:
            pass
    collection = chroma_client.get_or_create_collection(
        name=loc["collection"],
        metadata={
            "description": f"OEWN 2024 {pos_name} synset embeddings",
            "embedding_model": embedder_config.model,
            "source": "Open English WordNet 2024",
            "pos": pos_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    total = len(documents)
    stored = 0
    for doc_batch, text_batch in zip(
        chunked(documents, index_config.build_batch_size),
        chunked(embedding_texts, index_config.build_batch_size),
    ):
        embeddings = embedder.embed_batch(text_batch, kind="document")
        collection.add(
            ids=[doc["synset_id"] for doc in doc_batch],
            embeddings=embeddings,
            metadatas=[metadata_for_doc(doc) for doc in doc_batch],
            documents=text_batch,
        )
        stored += len(doc_batch)
        print(f"  Stored {stored:>6,} / {total:,}")
        if stored < total:
            time.sleep(index_config.rate_limit_delay)


def build_indexes(index_config: IndexConfig, embedder_config: EmbedderConfig) -> None:
    for pos in pos_values(index_config.pos):
        build_index_for_pos(pos, index_config, embedder_config)
