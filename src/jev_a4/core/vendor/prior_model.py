"""A4 fresh direct task head; no historical training or checkpoint entry points."""
from __future__ import annotations


def _sdpa(q, k, v, key_mask):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=key_mask[:, None, None, :], dropout_p=0.0, is_causal=False)


def make_prior_model(cfg, prior_feature_width):
    import torch
    from torch import nn

    width = int(cfg["decoder_width"])
    heads = int(cfg["attention_heads"])
    if width <= 0 or heads <= 0 or width % heads:
        raise ValueError("decoder width must be positive and divisible by attention heads")
    if int(prior_feature_width) < 4:
        raise ValueError("evidence requires class proportions and an availability flag")
    head_dim = width // heads
    hidden = int(cfg["backbone_hidden_width"])

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(width, width)
            self.k = nn.Linear(width, width)
            self.v = nn.Linear(width, width)
            self.o = nn.Linear(width, width)

        def forward(self, query, memory, memory_mask):
            batch, qlen, _ = query.shape
            mlen = memory.shape[1]
            q = self.q(query).view(batch, qlen, heads, head_dim).transpose(1, 2)
            k = self.k(memory).view(batch, mlen, heads, head_dim).transpose(1, 2)
            v = self.v(memory).view(batch, mlen, heads, head_dim).transpose(1, 2)
            value = _sdpa(q, k, v, memory_mask).transpose(1, 2).reshape(batch, qlen, width)
            return self.o(value)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(width)
            self.n2 = nn.LayerNorm(width)
            self.n3 = nn.LayerNorm(width)
            self.self_attn = Attention()
            self.cross_attn = Attention()
            self.ff = nn.Sequential(nn.Linear(width, int(cfg["ffn_width"])), nn.GELU(),
                                    nn.Dropout(float(cfg["dropout"])),
                                    nn.Linear(int(cfg["ffn_width"]), width))
            self.drop = nn.Dropout(float(cfg["dropout"]))

        def forward(self, query, query_mask, condition, condition_mask):
            query = query + self.drop(self.self_attn(self.n1(query), self.n1(query), query_mask))
            query = query * query_mask[:, :, None]
            query = query + self.drop(self.cross_attn(self.n2(query), condition, condition_mask))
            query = query * query_mask[:, :, None]
            query = query + self.drop(self.ff(self.n3(query)))
            return query * query_mask[:, :, None]

    class PriorSharedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_projection = nn.Linear(hidden, width)
            self.query_projection = nn.Linear(hidden, width)
            self.prior_projection = nn.Linear(hidden, width)
            self.condition_numeric = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            self.query_numeric = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            self.prior_numeric = nn.Sequential(nn.Linear(prior_feature_width, width), nn.GELU(),
                                               nn.Linear(width, width))
            self.condition_role = nn.Parameter(torch.empty(width))
            self.query_role = nn.Parameter(torch.empty(width))
            self.prior_role = nn.Parameter(torch.empty(width))
            self.readout = nn.Parameter(torch.empty(width))
            self.blocks = nn.ModuleList(Block() for _ in range(int(cfg["decoder_blocks"])))
            self.final_norm = nn.LayerNorm(width)
            self.joint_projection = nn.Linear(hidden, width)
            self.joint_role = nn.Parameter(torch.empty(width))
            self.joint_attention = Attention()
            self.changed_head = nn.Linear(width, 1)
            self.direction_head = nn.Linear(width, 1)
            for parameter in (self.condition_role, self.query_role, self.prior_role,
                              self.joint_role, self.readout):
                nn.init.normal_(parameter, std=0.02)
            # Preserve the selected A4 initialization: the fresh internal
            # joint-attention output projection starts at zero. It learns
            # from TRAIN labels; no earlier head or prediction is loaded.
            nn.init.zeros_(self.joint_attention.o.weight)
            nn.init.zeros_(self.joint_attention.o.bias)

        def forward(self, condition, condition_mask, condition_num, query, query_mask, query_num,
                    prior, prior_mask, prior_num, condition_inverse=None,
                    joint=None, joint_mask=None, joint_prior_tokens=None, prior_dropout=0.0):
            if condition_inverse is None:
                condition_inverse = torch.arange(condition.shape[0], device=condition.device)
            batch = len(query)
            if self.training and prior_dropout > 0:
                keep = torch.rand((batch,), device=query.device) >= float(prior_dropout)
                prior_mask = prior_mask & keep[:, None]
                if joint_mask is not None and joint_prior_tokens:
                    joint_mask = joint_mask.clone()
                    joint_mask[:, -int(joint_prior_tokens):] &= keep[:, None]
                neutral = torch.zeros_like(prior_num)
                neutral[:, :3] = 1.0 / 3.0
                # The final feature is the explicit evidence-available flag.
                neutral[:, -1] = 0.0
                prior_num = torch.where(keep[:, None], prior_num, neutral)

            c = self.condition_projection(condition) + self.condition_role
            cnum = self.condition_numeric(condition_num)[:, None, :]
            c = torch.cat([cnum, c], dim=1)
            cmask = torch.cat([torch.ones((len(condition), 1), device=condition.device, dtype=torch.bool),
                               condition_mask], dim=1)
            c = c[condition_inverse]
            cmask = cmask[condition_inverse]

            q = self.query_projection(query) + self.query_role
            p = self.prior_projection(prior) + self.prior_role
            qnum = self.query_numeric(query_num)[:, None, :]
            pnum = self.prior_numeric(prior_num)[:, None, :]
            readout = self.readout[None, None, :].expand(batch, 1, -1)
            q = torch.cat([qnum, q, pnum, p, readout], dim=1)
            qmask = torch.cat([
                torch.ones((batch, 1), device=query.device, dtype=torch.bool), query_mask,
                torch.ones((batch, 1), device=query.device, dtype=torch.bool), prior_mask,
                torch.ones((batch, 1), device=query.device, dtype=torch.bool)], dim=1)
            # Padded batches keep fixed-width query/card fields.  The readout is
            # physically the final token, irrespective of each row's true lengths.
            read_index = torch.full((batch,), q.shape[1] - 1, device=q.device, dtype=torch.long)
            for block in self.blocks:
                q = block(q, qmask, c, cmask)
            state = self.final_norm(q[torch.arange(batch, device=q.device), read_index])
            if joint is not None:
                joint = self.joint_projection(joint) + self.joint_role
                state = state + self.joint_attention(state[:, None, :], joint, joint_mask)[:, 0, :]
            return self.changed_head(state).squeeze(-1).float(), self.direction_head(state).squeeze(-1).float()

        @staticmethod
        def probabilities(changed, direction):
            changed_probability = torch.sigmoid(changed)
            up_given_changed = torch.sigmoid(direction)
            return torch.stack([changed_probability * up_given_changed,
                                1.0 - changed_probability,
                                changed_probability * (1.0 - up_given_changed)], dim=1)

    return PriorSharedHead()
