from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Protocol

import numpy as np

from .config import EmbedderConfig, env_value
from .utils import l2_normalize


class Embedder(Protocol):
    def embed(self, text: str, *, kind: str = "query") -> List[float]:
        ...

    def embed_batch(self, texts: List[str], *, kind: str = "document") -> List[List[float]]:
        ...


class APIEmbedder:
    def __init__(self, config: EmbedderConfig):
        self.config = config
        api_key = env_value(config.api_key_env, config.api_key)
        if not api_key:
            raise ValueError(f"Missing embedding API key. Set {config.api_key_env}.")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use API embedding providers.") from exc
        self.client = OpenAI(**kwargs)

    def _prefix(self, text: str, kind: str) -> str:
        if kind == "query" and self.config.query_prefix:
            return self.config.query_prefix + text
        if kind == "document" and self.config.document_prefix:
            return self.config.document_prefix + text
        return self.config.prefix + text if self.config.prefix else text

    def embed_batch(self, texts: List[str], *, kind: str = "document") -> List[List[float]]:
        inputs = [self._prefix(text, kind) for text in texts]
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(input=inputs, model=self.config.model)
                vectors = [item.embedding for item in response.data]
                if self.config.normalize:
                    return [l2_normalize(np.array(v, dtype=np.float32)).tolist() for v in vectors]
                return vectors
            except Exception as exc:
                if attempt >= self.config.max_retries - 1:
                    raise
                print(f"    Retry {attempt + 1}/{self.config.max_retries} after embedding error: {exc}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        return []

    def embed(self, text: str, *, kind: str = "query") -> List[float]:
        return self.embed_batch([text], kind=kind)[0]


class LocalSentenceTransformerEmbedder:
    def __init__(self, config: EmbedderConfig):
        self.config = config
        if config.auto_processor_text_only:
            try:
                from transformers import AutoProcessor, AutoTokenizer

                AutoProcessor.from_pretrained = AutoTokenizer.from_pretrained
            except ImportError as exc:
                raise RuntimeError("Install transformers to use auto_processor_text_only.") from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to use local embedding models.") from exc
        init_kwargs: Dict[str, Any] = dict(config.sentence_transformer_kwargs)
        if config.model_kwargs:
            init_kwargs["model_kwargs"] = config.model_kwargs
        if config.tokenizer_kwargs:
            init_kwargs["tokenizer_kwargs"] = config.tokenizer_kwargs
        self.model = SentenceTransformer(
            config.model,
            trust_remote_code=config.trust_remote_code,
            device=config.device,
            **init_kwargs,
        )

    def _prefix(self, text: str, kind: str) -> str:
        if kind == "query" and self.config.query_prefix:
            return self.config.query_prefix + text
        if kind == "document" and self.config.document_prefix:
            return self.config.document_prefix + text
        return self.config.prefix + text if self.config.prefix else text

    def embed_batch(self, texts: List[str], *, kind: str = "document") -> List[List[float]]:
        inputs = [self._prefix(text, kind) for text in texts]
        kwargs: Dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize,
            "show_progress_bar": len(inputs) > self.config.batch_size,
            **self.config.encode_kwargs,
        }
        prompt = self.config.query_prompt if kind == "query" else self.config.document_prompt
        prompt_name = self.config.query_prompt_name if kind == "query" else self.config.document_prompt_name
        task = self.config.query_task if kind == "query" else self.config.document_task
        if prompt or self.config.prompt:
            kwargs["prompt"] = prompt or self.config.prompt
        if prompt_name or self.config.prompt_name:
            kwargs["prompt_name"] = prompt_name or self.config.prompt_name
        if task or self.config.task:
            kwargs["task"] = task or self.config.task
        vectors = self.model.encode(inputs, **kwargs)
        return np.asarray(vectors).tolist()

    def embed(self, text: str, *, kind: str = "query") -> List[float]:
        return self.embed_batch([text], kind=kind)[0]


def create_embedder(config: EmbedderConfig) -> Embedder:
    provider = config.provider.strip().lower()
    if provider in {"openai", "cohere", "api", "openai-compatible"}:
        return APIEmbedder(config)
    if provider in {"local", "sentence-transformers", "sentence_transformers"}:
        return LocalSentenceTransformerEmbedder(config)
    raise ValueError(f"Unsupported embedder provider '{config.provider}'.")


def chunked(values: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
