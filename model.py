"""
model.py — GPT-2 wrapper with manual token-by-token generation.

Exposes `past_key_values` at every step so the caller can route
each KV-cache block through the SimulatedStorage tier system.
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ── helpers ──────────────────────────────────────────────────────────

def _pick_device() -> torch.device:
    """MPS if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── public API ───────────────────────────────────────────────────────

def load_model(device: torch.device | None = None):
    """Load gpt2 model + tokenizer onto *device*."""
    if device is None:
        device = _pick_device()
    print(f"[model] Loading GPT-2 on {device} …")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    print(f"[model] Ready  ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer, device


def generate_tokens(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    device: torch.device | None = None,
):
    """
    Yield one token at a time together with the full ``past_key_values``
    cache so the caller can feed blocks into SimulatedStorage.

    Yields
    ------
    step : int          (0-based generation step)
    token_id : int      (the chosen token)
    past_key_values     (tuple of (key, value) per layer)
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]          # (batch, vocab)
        next_token = torch.argmax(logits, dim=-1)   # greedy

        past_key_values = outputs.past_key_values

        # Print shape once on the first step
        if step == 0:
            print("\n[model] === KV-Cache shape (per layer) ===")
            k0, v0 = past_key_values[0]
            print(f"  key  : {k0.shape}   (batch, heads, seq_len, head_dim)")
            print(f"  value: {v0.shape}")
            print(f"  layers: {len(past_key_values)}\n")

        yield step, next_token.item(), past_key_values

        # Next iteration only needs the new token
        input_ids = next_token.unsqueeze(0)


# ── quick self-test ──────────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer, device = load_model()
    from kv_cache_rl import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT
    tokens = []

    for step, tok, past in generate_tokens(model, tokenizer, prompt, 20, device):
        tokens.append(tok)
        if step < 3 or step % 10 == 0:
            seq_len = past[0][0].shape[2]
            print(f"  step {step:>3d}  token={tokenizer.decode([tok])!r:>12s}  cache_seq_len={seq_len}")

    print("\n" + tokenizer.decode(
        tokenizer.encode(prompt) + tokens
    ))
