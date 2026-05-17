from __future__ import annotations

import ast
import json
from typing import Any, List, Tuple

import numpy as np


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content", ""))
                if isinstance(text, dict):
                    text = text.get("value", "")
                parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                parts.append(str(text if text is not None else item))
        return "".join(parts)
    return str(content)


def safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty JSON content")
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_synset_id(synset_id: str) -> str:
    synset_id = (synset_id or "").strip().lower()
    if synset_id.startswith("oewn-"):
        synset_id = synset_id[5:]
    parts = synset_id.split()
    return parts[0] if parts else synset_id


def parse_ground_truth(value: Any) -> Tuple[List[str], List[str]]:
    if value is None:
        return ([], [])
    raw = str(value).strip()
    if not raw:
        return ([], [])

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = ast.literal_eval(raw)
        items = [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]
    else:
        items = [raw]

    gold_ids: List[str] = []
    gold_glosses: List[str] = []
    for item in items:
        parts = item.strip().split(maxsplit=1)
        synset_id = normalize_synset_id(parts[0]) if parts else ""
        if synset_id:
            gold_ids.append(synset_id)
            gold_glosses.append(parts[1] if len(parts) > 1 else "")
    return gold_ids, gold_glosses


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm < eps else vec / norm


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def split_lemmas(lemmas: str) -> List[str]:
    text = (lemmas or "").strip().lower()
    if not text:
        return []
    for sep in [";", "|", "/"]:
        text = text.replace(sep, ",")
    return [" ".join(part.split()) for part in text.split(",") if part.strip()]
