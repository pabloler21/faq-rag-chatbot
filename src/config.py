"""Single source of configuration. os.getenv() must not appear anywhere else."""
import os
from pathlib import Path
from typing import Literal, NamedTuple

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

Role = Literal["llm", "embed", "judge"]


class _Endpoint(NamedTuple):
    """Which env vars carry the URL and key for one role, and their fallbacks."""

    url_var: str
    url_default: str
    key_var: str
    key_default: str


_ENDPOINTS: dict[Role, _Endpoint] = {
    "llm": _Endpoint(
        url_var="LLM_BASE_URL",
        url_default="http://localhost:1234/v1",
        key_var="LLM_API_KEY",
        key_default="lm-studio",
    ),
    "embed": _Endpoint(
        url_var="EMBED_BASE_URL",
        url_default="http://localhost:1234/v1",
        key_var="EMBED_API_KEY",
        key_default="lm-studio",
    ),
    "judge": _Endpoint(
        url_var="JUDGE_BASE_URL",
        url_default="https://api.openai.com/v1",
        key_var="OPENAI_API_KEY",
        key_default="",
    ),
}


def endpoint_url(role: Role) -> str:
    """The URL a role actually talks to, for error messages."""
    endpoint = _ENDPOINTS[role]
    return os.getenv(endpoint.url_var, endpoint.url_default)


def unreachable(role: Role) -> RuntimeError:
    """Turn httpx connection noise into an actionable message."""
    url = endpoint_url(role)

    if "localhost" in url:
        hint = "Is LM Studio running with the server started?"
    else:
        hint = "Check your network."

    return RuntimeError(f"Cannot reach the '{role}' endpoint at {url}. {hint}")


def get_client(role: Role) -> OpenAI:
    """Return an OpenAI-compatible client for the given role."""
    endpoint = _ENDPOINTS[role]
    api_key = os.getenv(endpoint.key_var, endpoint.key_default)

    if not api_key:
        raise RuntimeError(f"Missing {endpoint.key_var}. Set it in your .env file.")

    return OpenAI(base_url=endpoint_url(role), api_key=api_key)
