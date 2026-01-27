"""
KV-Cache enabled model for efficient inference.
"""

import math
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers.activations
from torch.nn import functional as F
from transformers import GPT2Config

ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "past_key_values"])
KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key, value) per layer


class CausalSelfAttentionKV(nn.Module):

    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,
            )
        self.attention_weights = attention_weights

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Use past key-values if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, nh, past_T + T, hs)
            v = torch.cat([past_v, v], dim=2)  # (B, nh, past_T + T, hs)

        present_kv = (k, v) if use_cache else None

        full_T = k.size(2)  # total sequence length including past

        # causal self-attention
        if self.flash and self.attention_weights is None:
            # efficient attention using Flash Attention CUDA kernels
            if past_kv is not None:
                # Disable is_causal since we're using cache
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False, 
                )
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Adjust mask for KV cache - we only compute attention for new tokens
            att = att.masked_fill(self.bias[:, :, full_T-T:full_T, :full_T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            if self.attention_weights is not None:
                self.attention_weights.append(att.detach().cpu())

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv


class BlockKV(nn.Module):
    """Transformer block with KV caching support."""

    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionKV(config, attention_weights=attention_weights)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        # Import MLP from original model
        from .model import MLP
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT2LMNoBiasModelKV(nn.Module):
    """GPT2 model with KV caching support for efficient inference."""

    def __init__(
        self,
        config: GPT2Config,
        return_attention=False,
    ):
        super().__init__()
        self.config = config

        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.n_positions, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(
                    [BlockKV(config, self.attention_weights) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        pos = torch.arange(0, config.n_positions, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @lru_cache
    def num_parameters(self, exclude_embeddings=True):
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[KVCache, ...]] = None,
        use_cache: bool = False
    ) -> ModelOutput:
        _, t = input_ids.size()
        if self.return_attention:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)
        # Always use positions starting from 0 (critical to maintain positional encoding)
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            pos_emb = self.transformer.wpe(self.pos[past_length:past_length + t])
        else:
            pos_emb = self.transformer.wpe(self.pos[:t])
        x = self.transformer.drop(tok_emb + pos_emb)

        present_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                present_key_values.append(present_kv)

        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            logits = self.lm_head(x)
            loss = None

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=tuple(present_key_values) if use_cache else None
        )

    @torch.no_grad()
    def get_next_token(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
        top_k: int | None = None,
        past_key_values: Optional[Tuple[KVCache, ...]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[KVCache, ...]]]:

        # only process last token when we have a cache
        if past_key_values is not None:
            x = x[:, -1:]

        output = self(x, past_key_values=past_key_values, use_cache=use_cache)
        logits = output.logits[:, -1, :]

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if return_probs:
            return next_token, probs, output.past_key_values
        return next_token, None, output.past_key_values


def convert_pretrained_to_kv_model(pretrained_model) -> GPT2LMNoBiasModelKV:

    config = pretrained_model.config
    kv_model = GPT2LMNoBiasModelKV(config, return_attention=pretrained_model.return_attention)

    # copy embeddings and head
    kv_model.transformer.wte.weight = pretrained_model.transformer.wte.weight
    kv_model.transformer.wpe.weight = pretrained_model.transformer.wpe.weight
    kv_model.lm_head.weight = pretrained_model.lm_head.weight

    # Copy dropout
    kv_model.transformer.drop = pretrained_model.transformer.drop

    # copy layer norm
    kv_model.transformer.ln_f.weight = pretrained_model.transformer.ln_f.weight
    kv_model.transformer.ln_f.bias = pretrained_model.transformer.ln_f.bias

    # Copy each transformer block
    for kv_block, orig_block in zip(kv_model.transformer.h, pretrained_model.transformer.h):
        # Copy attention
        kv_block.attn.c_attn.weight = orig_block.attn.c_attn.weight
        if orig_block.attn.c_attn.bias is not None:
            kv_block.attn.c_attn.bias = orig_block.attn.c_attn.bias
        kv_block.attn.c_proj.weight = orig_block.attn.c_proj.weight
        if orig_block.attn.c_proj.bias is not None:
            kv_block.attn.c_proj.bias = orig_block.attn.c_proj.bias
        kv_block.attn.attn_dropout = orig_block.attn.attn_dropout
        kv_block.attn.resid_dropout = orig_block.attn.resid_dropout

        # Copy layer norms
        kv_block.ln_1.weight = orig_block.ln_1.weight
        kv_block.ln_1.bias = orig_block.ln_1.bias
        kv_block.ln_2.weight = orig_block.ln_2.weight
        kv_block.ln_2.bias = orig_block.ln_2.bias

        # Copy MLP
        kv_block.mlp.c_fc.weight = orig_block.mlp.c_fc.weight
        if orig_block.mlp.c_fc.bias is not None:
            kv_block.mlp.c_fc.bias = orig_block.mlp.c_fc.bias
        kv_block.mlp.c_proj.weight = orig_block.mlp.c_proj.weight
        if orig_block.mlp.c_proj.bias is not None:
            kv_block.mlp.c_proj.bias = orig_block.mlp.c_proj.bias
        kv_block.mlp.dropout = orig_block.mlp.dropout

    return kv_model