from typing import List, Dict
from openai import OpenAI
import requests

import config



OPENAI_API_KEY = config.get_config("LLM","OPENAI_API_KEY", fallback=None)
OPENAI_MODEL = config.get_config("LLM","OPENAI_MODEL", fallback=None)
USE_OLLAMA = config.get_config("LLM","USE_OLLAMA", fallback="0") == "1"
OLLAMA_MODEL = config.get_config("LLM","OLLAMA_MODEL", fallback=None)

def llm_complete_openai(messages: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Set it or enable Ollama.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()
    # return "OpenAI Response"


def llm_complete_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
    # return "Ollama Response"

def generate_response(prompt: str) -> str:
    if USE_OLLAMA:
        return llm_complete_ollama(prompt)
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return llm_complete_openai(messages)