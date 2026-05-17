from __future__ import annotations

import json
from typing import Dict


def build_translation_system_prompt(language: str) -> str:
    return f"""You are an expert translator specializing in {language}.

# TASK:
Translate the given sentence into English in TWO ways:
1. a literal translation that stays close to the source wording
2. a natural translation that sounds fluent in modern English

Focus on correctly representing the literal and metaphorical meanings of specific words.

# OUTPUT:
Return ONLY valid JSON with exactly these keys:
{{
  "literal": "string",
  "natural": "string"
}}

No commentary, no markdown, no extra keys.
"""


def build_enriched_gloss_system_prompt(language: str) -> str:
    return f"""You are an expert lexicographer and linguist specializing in {language} semantics.

# TASK:
You will be given:
- a target token
- its dictionary lemma
- the original sentence it occurs in
- optionally, two proposed English translations of the sentence.

Using all of this context, produce:
1. 1 to 3 possible English dictionary-style definitions of the target word in this context,
   ordered by likelihood (most likely first). Each definition must be a phrase that fully captures and explains the sense. Example: "she ran all the way home" -> "move rapidly from one place to another"
2. 1 to 5 candidate English lemmas or short expressions (1-2 words, e.g. "make up", "go out")
   that could translate the token in this context.

# GUIDELINES:
- Be specific and detailed enough to distinguish senses.
- Account for negation: define the word's meaning, not its truth value. Example: "she didn't run" -> "move at a speed faster than a walk", not "stand still".
- Account for metaphorical meanings: when in doubt, include both literal and metaphorical definitions. Example: "she saw a risk in his plan" -> "perceive a situation mentally" works better than "perceive by sight".
- Beware of distinguishing the actual meaning of the target word from those of its arguments.
- Avoid unnecessary contextual information unless it is part of the lexicalized sense.
- If there is genuine ambiguity, include multiple definitions; otherwise output 1.
- Keep outputs precise and consistent.

OUTPUT:
Return ONLY valid JSON with exactly these keys:
{{
  "definitions": ["def1", "def2", "def3"],
  "candidate_lemmas": ["lemma1", "lemma2", "lemma3", "lemma4", "lemma5"]
}}

No commentary, no markdown, no extra keys.
"""


def build_gloss_user_prompt(
    token: str,
    lemma: str,
    sentence: str,
    translations: Dict[str, str],
    pos: str,
) -> str:
    return (
        f"PartOfSpeech: {pos}\n"
        f"Token: {token}\n"
        f"Lemma: {lemma}\n"
        f"Sentence: {sentence}\n"
        f"TranslationsJSON: {json.dumps(translations or {}, ensure_ascii=False)}"
    )
