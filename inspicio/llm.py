from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .config import LLMConfig, env_value
from .prompts import build_enriched_gloss_system_prompt, build_gloss_user_prompt, build_translation_system_prompt
from .utils import message_content_to_text, safe_json_loads


DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai-compatible": None,
}


class ChatLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        provider = config.provider.strip().lower()
        base_url = config.base_url or DEFAULT_BASE_URLS.get(provider)
        api_key = env_value(config.api_key_env, config.api_key)
        if not api_key:
            raise ValueError(f"Missing API key for LLM provider '{config.provider}'. Set {config.api_key_env}.")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if config.default_headers:
            kwargs["default_headers"] = config.default_headers
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use API LLM providers.") from exc
        self.client = OpenAI(**kwargs)

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            **self.config.extra_params,
        }
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                text = message_content_to_text(response.choices[0].message.content)
                return safe_json_loads(text)
            except Exception as exc:
                if attempt >= self.config.max_retries - 1:
                    raise
                print(f"    Retry {attempt + 1}/{self.config.max_retries} after LLM error: {exc}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        raise RuntimeError("LLM request failed")

    def translate_sentence(self, language: str, sentence: str, temperature: float) -> Dict[str, str]:
        obj = self.chat_json(
            build_translation_system_prompt(language),
            f"Sentence: {sentence}",
            temperature=temperature,
        )
        return {
            "literal": str(obj.get("literal") or obj.get("Literal") or "").strip(),
            "natural": str(obj.get("natural") or obj.get("Natural") or "").strip(),
        }

    def generate_enriched_gloss(
        self,
        language: str,
        token: str,
        lemma: str,
        sentence: str,
        translations: Optional[Dict[str, str]],
        pos: str,
        temperature: float,
        max_definitions: int,
        max_lemmas: int,
    ) -> Dict[str, Any]:
        obj = self.chat_json(
            build_enriched_gloss_system_prompt(language),
            build_gloss_user_prompt(token, lemma, sentence, translations or {}, pos),
            temperature=temperature,
        )
        definitions = obj.get("definitions") or obj.get("glosses") or []
        lemmas = obj.get("candidate_lemmas") or obj.get("lemmas") or []
        if not isinstance(definitions, list):
            definitions = [str(definitions)]
        if not isinstance(lemmas, list):
            lemmas = [str(lemmas)]
        return {
            "definitions": [str(x).strip() for x in definitions if str(x).strip()][:max_definitions],
            "candidate_lemmas": [str(x).strip() for x in lemmas if str(x).strip()][:max_lemmas],
        }
