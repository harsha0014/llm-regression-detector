import os
import time
import json
from openai import AsyncOpenAI
from src.models import PromptConfig

# Global variable to hold the client singleton
_client = None

def get_client():
    """Lazily create the OpenAI client only when needed."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Make sure .env file is loaded or set it in your shell."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client

async def classify_email(email: str, config: PromptConfig) -> dict:
    """
    Call the LLM with the versioned prompt and return structured JSON.
    """
    # Build the user message with few-shot examples if any
    user_content = "Classify the following customer support email and provide a one‑sentence summary.\n\n"
    if config.few_shot_examples:
        user_content += "Here are some examples:\n"
        for ex in config.few_shot_examples:
            user_content += f"Email: {ex['email']}\n"
            user_content += f"Category: {ex['category']}\n"
            user_content += f"Summary: {ex['summary']}\n\n"
    user_content += f"Email: {email}\n"
    user_content += "Return valid JSON with keys: category, summary."

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content}
    ]

    client = get_client()  # <-- Client is created HERE, after .env is loaded
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        latency = time.perf_counter() - start
        raw = response.choices[0].message.content
        data = json.loads(raw)
        category = data.get("category", "unknown")
        summary = data.get("summary", "")
        usage = response.usage
        return {
            "category": category,
            "summary": summary,
            "latency_ms": latency * 1000,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "category": "error",
            "summary": "",
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
        }