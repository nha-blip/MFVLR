"""Tokenizer abstraction and boundary for MFVLR language prompt representation.

PAPER_SPECIFIED:
- Sequence length: n = 308 tokens.
- Vocabulary size: s = 49,408 tokens.
- Every token ID must satisfy: 0 <= token_id < 49,408.

ASSUMPTION_FROM_PAPER_GAP:
- Exact BPE tokenizer vocabulary and merge rules are not explicitly published with the paper.
- A deterministic, offline CLIP-compatible tokenization algorithm with vocabulary size 49,408 is used.
- No network requests or online downloads are performed.
"""

from typing import List, Optional, Union
import hashlib
import torch


class MFVLRTokenizer:
    """Self-contained deterministic tokenizer interface for MFVLR.

    # PAPER_SPECIFIED:
    # Output shape: [308] tokens with values in [0, 49407].
    #
    # ASSUMPTION_FROM_PAPER_GAP:
    # Offline deterministic token encoding mapping UTF-8 tokens into vocabulary space [0, 49407].
    """

    def __init__(
        self,
        max_tokens: int = 308,
        vocab_size: int = 49408,
        bos_id: int = 49406,
        eos_id: int = 49407,
        pad_id: int = 0,
    ):
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode(self, text: str) -> torch.Tensor:
        """Encode textual prompt to exactly 308 token IDs.

        Args:
            text: Input prompt text string.

        Returns:
            torch.Tensor of shape [308] and dtype torch.long with IDs in [0, 49407].
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected prompt text to be str, got {type(text).__name__}")

        # Deterministic tokenization
        words = text.strip().split()
        token_ids: List[int] = [self.bos_id]

        for word in words:
            # Deterministic hash to map word string to token ID in valid vocabulary range [1, vocab_size - 3]
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            token_id = 1 + (h % (self.vocab_size - 3))
            token_ids.append(token_id)
            if len(token_ids) >= self.max_tokens - 1:
                break

        token_ids.append(self.eos_id)

        # Pad or truncate to exact length max_tokens (308)
        if len(token_ids) < self.max_tokens:
            token_ids = token_ids + [self.pad_id] * (self.max_tokens - len(token_ids))
        else:
            token_ids = token_ids[:self.max_tokens]

        tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Strict validation of contract
        if tokens_tensor.shape != (self.max_tokens,):
            raise ValueError(f"Expected token tensor of shape ({self.max_tokens},), got {tokens_tensor.shape}")
        if (tokens_tensor < 0).any() or (tokens_tensor >= self.vocab_size).any():
            raise ValueError(f"Token IDs out of valid range [0, {self.vocab_size - 1}]")

        return tokens_tensor

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)


_DEFAULT_TOKENIZER = MFVLRTokenizer()


def tokenize(text: str, max_tokens: int = 308, vocab_size: int = 49408) -> torch.Tensor:
    """Functional interface to tokenize prompt text.

    Args:
        text: Prompt text.
        max_tokens: Number of tokens (default: 308).
        vocab_size: Vocabulary size (default: 49408).

    Returns:
        torch.Tensor of shape [308] and dtype torch.long.
    """
    tokenizer = MFVLRTokenizer(max_tokens=max_tokens, vocab_size=vocab_size)
    return tokenizer.encode(text)
