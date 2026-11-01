"""Pinned, offline BGE-M3 token counting; no model weights or remote code."""

import gzip
import hashlib
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA256 = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    raw = gzip.decompress((Path(__file__).parent / "assets/bge-m3-tokenizer.json.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("BGE-M3 tokenizer checksum mismatch")
    result = Tokenizer.from_str(raw.decode())
    result.no_truncation()
    result.no_padding()
    return result


def token_count(value: str) -> int:
    """Include BOS/EOS in the embedding input budget."""
    return len(tokenizer().encode(value, add_special_tokens=True).ids)


def bounded(value: str, budget: int) -> str:
    """Return a Unicode-safe prefix within a token budget, including BOS/EOS."""
    if budget <= 2:
        return ""
    encoding = tokenizer().encode(value, add_special_tokens=False)
    if len(encoding.ids) + 2 <= budget:
        return value
    end = encoding.offsets[budget - 3][1]
    result = value[:end]
    # Several tokens can describe one Unicode codepoint; re-encoding the prefix
    # can also change its final token. Always verify the actual returned input.
    while result and token_count(result) > budget:
        result = result[:-1]
    return result
