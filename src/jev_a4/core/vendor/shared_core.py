"""Exact frozen Qwen3.5 contiguous-layer execution used by the A4 encoder."""
from __future__ import annotations


def _text_components(text_model, input_ids, attention_mask):
    """Reproduce Qwen3.5TextModel's position/mask setup for depth splitting."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        create_causal_mask, create_recurrent_attention_mask,
    )
    inputs_embeds = text_model.embed_tokens(input_ids)
    length = inputs_embeds.shape[1]
    position_ids = torch_arange(length, inputs_embeds.device).view(1, 1, -1).expand(
        4, inputs_embeds.shape[0], -1
    )
    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]
    kwargs = {
        "config": text_model.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**kwargs),
        "linear_attention": create_recurrent_attention_mask(**kwargs),
    }
    rotary = text_model.rotary_emb(inputs_embeds, rope_position_ids)
    return inputs_embeds, text_position_ids, rotary, masks


def torch_arange(length, device):
    import torch
    return torch.arange(length, device=device)


def split_forward(text_model, input_ids, attention_mask, start=0, end=None, hidden_states=None, final_norm=False):
    """Execute an exact contiguous Qwen3.5 layer interval without a fake inputs_embeds restart."""
    if end is None:
        end = text_model.config.num_hidden_layers
    embedded, position_ids, rotary, masks = _text_components(text_model, input_ids, attention_mask)
    hidden = embedded if hidden_states is None else hidden_states
    for i in range(start, end):
        hidden = text_model.layers[i](
            hidden,
            position_embeddings=rotary,
            attention_mask=masks[text_model.config.layer_types[i]],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
    if final_norm:
        hidden = text_model.norm(hidden)
    return hidden
