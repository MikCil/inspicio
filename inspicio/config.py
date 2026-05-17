from __future__ import annotations

import os
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


POS_ALIASES = {
    "v": "verb",
    "verb": "verb",
    "verbs": "verb",
    "n": "noun",
    "noun": "noun",
    "nouns": "noun",
    "a": "adjective",
    "adj": "adjective",
    "adjective": "adjective",
    "adjectives": "adjective",
    "r": "adverb",
    "adv": "adverb",
    "adverb": "adverb",
    "adverbs": "adverb",
}

POS_TO_WN = {
    "verb": "v",
    "noun": "n",
    "adjective": "a",
    "adverb": "r",
}

POS_PLURALS = {
    "verb": "verbs",
    "noun": "nouns",
    "adjective": "adjectives",
    "adverb": "adverbs",
}


def normalize_pos(value: Optional[str], default: str = "verb") -> str:
    key = (value or default or "verb").strip().lower()
    key = key.replace(" ", "_")
    if key not in POS_ALIASES:
        raise ValueError(f"Unsupported PoS '{value}'. Expected verb, noun, adjective, or adverb.")
    return POS_ALIASES[key]


def pos_values(value: str) -> List[str]:
    if (value or "").strip().lower() == "all":
        return ["verb", "noun", "adjective", "adverb"]
    return [normalize_pos(value)]


def env_value(name: Optional[str], explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if name:
        return os.environ.get(name, "")
    return ""


@dataclass
class ColumnConfig:
    id: str = "ID"
    language: str = "language"
    token: str = "TOKEN"
    lemma: str = "LEMMA"
    sentence: str = "SENTENCE"
    pos: str = "PoS"
    ground_truth: str = "VERB SEMANTICS"


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-reasoner"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str = ""
    base_url: Optional[str] = None
    default_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 5.0
    request_delay: float = 0.1
    max_tokens: Optional[int] = None


@dataclass
class EmbedderConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    base_url: Optional[str] = None
    batch_size: int = 100
    request_delay: float = 0.1
    max_retries: int = 3
    retry_delay: float = 5.0
    normalize: bool = True
    trust_remote_code: bool = False
    device: Optional[str] = None
    prefix: str = ""
    query_prefix: str = ""
    document_prefix: str = ""
    prompt: Optional[str] = None
    prompt_name: Optional[str] = None
    query_prompt: Optional[str] = None
    document_prompt: Optional[str] = None
    query_prompt_name: Optional[str] = None
    document_prompt_name: Optional[str] = None
    task: Optional[str] = None
    query_task: Optional[str] = None
    document_task: Optional[str] = None
    auto_processor_text_only: bool = False
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    sentence_transformer_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexConfig:
    lexicon_id: str = "oewn:2024"
    pos: str = "all"
    chroma_path_template: str = "./oewn_{pos_name}_embeddings"
    collection_template: str = "oewn_{pos_plural}"
    pos_indexes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    lemma_cache_template: str = "./lemma_index_{collection}.json"
    build_batch_size: int = 100
    store_batch_size: int = 500
    rate_limit_delay: float = 0.3
    reset_collection: bool = True

    def location_for_pos(self, pos: str) -> Dict[str, str]:
        pos_name = normalize_pos(pos)
        custom = self.pos_indexes.get(pos_name, {})
        values = {
            "pos": POS_TO_WN[pos_name],
            "pos_name": pos_name,
            "pos_plural": POS_PLURALS[pos_name],
        }
        return {
            "chroma_path": custom.get("chroma_path") or self.chroma_path_template.format(**values),
            "collection": custom.get("collection") or self.collection_template.format(**values),
            "lemma_cache": custom.get("lemma_cache"),
        }

    def lemma_cache_for(self, collection: str, pos: str) -> str:
        loc = self.location_for_pos(pos)
        if loc.get("lemma_cache"):
            return str(loc["lemma_cache"])
        return self.lemma_cache_template.format(collection=collection, pos_name=normalize_pos(pos))


@dataclass
class RetrievalConfig:
    top_k: int = 50
    dense_top_n_per_definition: int = 100
    definition_decay_weights: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    lemma_multiplier_gamma: float = 0.8
    lemma_only_default_score: float = 0.65
    normalize_final_scores: bool = False
    max_candidates_before_mmr: int = 500
    enable_mmr: bool = True
    mmr_lambda: float = 0.8
    enable_lexname_coverage: bool = False
    lexname_min_candidates: int = 1
    lemma_index_batch_size: int = 2000


@dataclass
class GenerationConfig:
    translation_enabled: bool = True
    translation_temperature: float = 1.3
    gloss_temperature: float = 0.8
    max_definitions: int = 3
    max_lemmas: int = 5


@dataclass
class RunConfig:
    input_csv: str = "input.csv"
    input_jsonl: Optional[str] = None
    output_jsonl: str = "outputs/output.jsonl"
    report_file: str = "outputs/evaluation.txt"
    mode: str = "full"
    resume: bool = True
    default_pos: str = "verb"
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _coerce_dataclass(cls, data: Mapping[str, Any]):
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in dict(data or {}).items() if k in allowed})


def load_raw_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text) or {}
    try:
        import yaml
    except ImportError as exc:
        return _load_simple_yaml(text)
    return yaml.safe_load(text) or {}


def _parse_simple_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_simple_scalar(value)
        parent[key.strip()] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def load_config(path: str | Path) -> RunConfig:
    raw = load_raw_config(path)
    cfg = RunConfig()
    top_level = {
        k: v
        for k, v in raw.items()
        if k
        in {
            "input_csv",
            "input_jsonl",
            "output_jsonl",
            "report_file",
            "mode",
            "resume",
            "default_pos",
        }
    }
    cfg = RunConfig(**{**cfg.__dict__, **top_level})
    cfg.columns = _coerce_dataclass(ColumnConfig, raw.get("columns", {}))
    cfg.llm = _coerce_dataclass(LLMConfig, raw.get("llm", {}))
    cfg.embedder = _coerce_dataclass(EmbedderConfig, raw.get("embedder", {}))
    cfg.index = _coerce_dataclass(IndexConfig, raw.get("index", {}))
    cfg.retrieval = _coerce_dataclass(RetrievalConfig, raw.get("retrieval", {}))
    cfg.generation = _coerce_dataclass(GenerationConfig, raw.get("generation", {}))
    cfg.default_pos = normalize_pos(cfg.default_pos)
    return cfg


def merge_cli_overrides(cfg: RunConfig, **overrides: Any) -> RunConfig:
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.default_pos:
        cfg.default_pos = normalize_pos(cfg.default_pos)
    return cfg


def ensure_parent(path: str | Path) -> None:
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def row_pos(row: Mapping[str, Any], columns: ColumnConfig, default: str) -> str:
    candidates: Iterable[str] = (columns.pos, "PoS", "POS", "pos", "part_of_speech")
    for name in candidates:
        value = row.get(name)
        if value:
            return normalize_pos(str(value), default)
    return normalize_pos(default)
