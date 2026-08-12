"""Single source of configuration. os.getenv() must not appear anywhere else."""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "faq_document.txt"
INDEX_DIR = ROOT / "index"
OUTPUTS_DIR = ROOT / "outputs"
EVAL_DIR = ROOT / "eval"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-4b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-nano")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.65"))
MIN_CHUNKS = 2          # the rubric requires returning 2-5 chunks, always

_ENDPOINTS = {
    "llm":   ("LLM_BASE_URL",   "http://localhost:1234/v1",  "LLM_API_KEY",    "lm-studio"),
    "embed": ("EMBED_BASE_URL", "http://localhost:1234/v1",  "EMBED_API_KEY",  "lm-studio"),
    "judge": ("JUDGE_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY", ""),
}


def endpoint_url(role: Literal["llm", "embed", "judge"]) -> str:
    """The URL a role actually talks to, for error messages."""
    url_var, url_default, _, _ = _ENDPOINTS[role]
    return os.getenv(url_var, url_default)


def unreachable(role: Literal["llm", "embed", "judge"]) -> RuntimeError:
    """Turn httpx connection noise into an actionable message."""
    url = endpoint_url(role)
    hint = "Is LM Studio running with the server started?" if "localhost" in url else "Check your network."
    return RuntimeError(f"Cannot reach the '{role}' endpoint at {url}. {hint}")


def get_client(role: Literal["llm", "embed", "judge"]) -> OpenAI:
    """Return an OpenAI-compatible client for the given role."""
    url_var, url_default, key_var, key_default = _ENDPOINTS[role]
    api_key = os.getenv(key_var, key_default)
    if not api_key:
        raise RuntimeError(f"Missing {key_var}. Set it in your .env file.")
    return OpenAI(base_url=os.getenv(url_var, url_default), api_key=api_key)
