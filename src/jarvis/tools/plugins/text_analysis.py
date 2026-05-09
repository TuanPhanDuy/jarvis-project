"""Plugin: analyze_text — local NLP analysis via Ollama structured prompts.

Supports: sentiment, entities, keywords, summarize, classify, language, batch.
All tasks run entirely on-device — no external APIs.
"""
from __future__ import annotations

import json


_TASK_PROMPTS = {
    "sentiment": (
        "Analyze the sentiment of the following text. "
        "Classify as POSITIVE, NEGATIVE, or NEUTRAL. "
        "Give a confidence score from 0.0 to 1.0. "
        'Reply with JSON only, no other text: {"sentiment": "POSITIVE", "score": 0.87, "reasoning": "..."}'
    ),
    "entities": (
        "Extract all named entities from the following text. "
        "Categories: PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, EVENT. "
        'Reply with JSON only: {"entities": [{"text": "...", "type": "PERSON"}, ...]}'
    ),
    "keywords": (
        "Extract the 5 to 10 most important keywords and key phrases from the following text. "
        'Reply with JSON only: {"keywords": ["keyword1", "phrase two", ...]}'
    ),
    "summarize": (
        "Summarize the following text in 2 to 3 concise sentences. "
        "Reply with the summary text only, no JSON, no preamble."
    ),
    "classify": (
        "Classify the topic and domain of the following text. "
        "Choose the most appropriate category (e.g. Technology, Science, Politics, "
        "Business, Health, Sports, Entertainment, Education, Other). "
        'Reply with JSON only: {"category": "Technology", "subcategory": "AI", "confidence": 0.92}'
    ),
    "language": (
        "Detect the language of the following text. "
        'Reply with JSON only: {"language": "English", "code": "en", "confidence": 0.99}'
    ),
}


def _call_ollama(prompt: str, text: str) -> str:
    import ollama
    from jarvis.config import get_settings
    model = get_settings().model
    full_prompt = f"{prompt}\n\nText to analyze:\n{text[:4000]}"
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        options={"temperature": 0.1},
    )
    return response.message.content.strip()


def _parse_json(raw: str) -> dict | list:
    start = raw.find("{") if "{" in raw else raw.find("[")
    end = raw.rfind("}") if "{" in raw else raw.rfind("]")
    if start == -1 or end == -1:
        return {}
    return json.loads(raw[start:end + 1])


def handle(tool_input: dict) -> str:
    try:
        text = str(tool_input.get("text", "")).strip()
        task = str(tool_input.get("task", "sentiment")).lower()

        if not text:
            return "ERROR: 'text' is required"

        if task == "batch":
            tasks = tool_input.get("tasks", ["sentiment", "keywords", "classify"])
            results = {}
            for t in tasks:
                if t in _TASK_PROMPTS:
                    raw = _call_ollama(_TASK_PROMPTS[t], text)
                    try:
                        results[t] = _parse_json(raw) if t != "summarize" else raw
                    except Exception:
                        results[t] = raw
            return json.dumps(results, indent=2, ensure_ascii=False)

        if task not in _TASK_PROMPTS:
            return (
                f"ERROR: unknown task '{task}'. "
                f"Valid: {', '.join(_TASK_PROMPTS)} , batch"
            )

        raw = _call_ollama(_TASK_PROMPTS[task], text)

        if task == "summarize":
            return f"Summary: {raw}"

        try:
            parsed = _parse_json(raw)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return raw

    except Exception as e:
        return f"ERROR: analyze_text failed — {e}"


SCHEMA: dict = {
    "name": "analyze_text",
    "description": (
        "Perform local NLP analysis on any text using the on-device language model. "
        "Tasks: 'sentiment' (POSITIVE/NEGATIVE/NEUTRAL + score), "
        "'entities' (named entity extraction: PERSON, ORG, LOCATION, DATE), "
        "'keywords' (top 5-10 keywords/phrases), "
        "'summarize' (2-3 sentence summary), "
        "'classify' (topic/domain classification), "
        "'language' (language detection), "
        "'batch' (run multiple tasks at once via the 'tasks' parameter)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze (up to 4000 characters).",
            },
            "task": {
                "type": "string",
                "enum": ["sentiment", "entities", "keywords", "summarize", "classify", "language", "batch"],
                "description": "Which analysis to run. Default: 'sentiment'.",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For 'batch' task only: list of tasks to run simultaneously.",
            },
        },
        "required": ["text"],
    },
}
