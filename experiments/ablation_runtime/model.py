"""Exact fresh A4 head and physical leave-one-input-branch ablations.

All common parameters are initialized in the original FULL order before the
ablated modules are deleted. No task checkpoint or old prediction is read.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

from experiments.ablation_runtime.contract import VARIANTS, roles


def _factory():
    from jev_a4.core.vendor import prior_model
    return prior_model


def make_prior_model(cfg, prior_feature_width):
    import torch

    variant = cfg.get('branch_variant', 'FULL')
    if variant not in VARIANTS:
        raise ValueError(f'unknown complete branch ablation {variant}')
    head = _factory().make_prior_model(cfg, prior_feature_width)
    head.branch_variant = variant
    head.prior_feature_width = int(prior_feature_width)
    head.full_initialized_parameter_count = sum(p.numel() for p in head.parameters())
    head.joint_enabled = cfg.get('joint_enabled', True)
    head.flat_output = cfg.get('flat_output', False)
    if not cfg.get('numeric_enabled',True):
        class ZeroNumeric(torch.nn.Module):
            def __init__(self, width):
                super().__init__(); self.width = width
            def forward(self, values):
                return values.new_zeros((len(values),self.width))
        # Zero the whole MLP output (including biases), not just its input.
        for name in ('condition_numeric','query_numeric','prior_numeric'):
            setattr(head,name,ZeroNumeric(int(cfg['decoder_width'])))
    excluded = [name for name, keep in zip(('condition','query','prior'), roles(variant)) if not keep]
    for branch in excluded:
        for name in (branch + '_projection', branch + '_numeric', branch + '_role'):
            delattr(head, name)
        if branch == 'condition':
            for block in head.blocks:
                del block.cross_attn
                del block.n2
    if not head.joint_enabled:
        for name in ('joint_projection','joint_role','joint_attention'):
            delattr(head, name)
    if head.flat_output:
        head.flat_head = torch.nn.Linear(int(cfg['decoder_width']), 3)
        del head.changed_head
        del head.direction_head
    if excluded or not head.joint_enabled or head.flat_output:

        def ablated_forward(self, condition, condition_mask, condition_num,
                            query, query_mask, query_num, prior, prior_mask,
                            prior_num, condition_inverse=None, joint=None,
                            joint_mask=None, joint_prior_tokens=None, prior_dropout=0.0):
            # Placeholder shapes are legal but excluded tensors are never
            # numerically read, projected, or converted into a constant token.
            batch = query.shape[0]
            device = self.readout.device
            use_condition, use_query, use_prior = roles(self.branch_variant)
            if prior_dropout:
                raise ValueError('authoritative pre-last8 dropout belongs to Partition.batch, not the task head')
            c = cmask = None
            if use_condition:
                if condition_inverse is None:
                    condition_inverse = torch.arange(len(condition), device=device)
                c = self.condition_projection(condition) + self.condition_role
                c = torch.cat([self.condition_numeric(condition_num)[:, None, :], c], dim=1)
                cmask = torch.cat([torch.ones((len(condition), 1), dtype=torch.bool, device=device),
                                   condition_mask], dim=1)
                c, cmask = c[condition_inverse], cmask[condition_inverse]
            tokens, masks = [], []
            if use_query:
                tokens.extend([self.query_numeric(query_num)[:, None, :],
                               self.query_projection(query) + self.query_role])
                masks.extend([torch.ones((batch, 1), dtype=torch.bool, device=device), query_mask])
            if use_prior:
                tokens.extend([self.prior_numeric(prior_num)[:, None, :],
                               self.prior_projection(prior) + self.prior_role])
                masks.extend([torch.ones((batch, 1), dtype=torch.bool, device=device), prior_mask])
            tokens.append(self.readout[None, None, :].expand(batch, 1, -1))
            masks.append(torch.ones((batch, 1), dtype=torch.bool, device=device))
            q, qmask = torch.cat(tokens, dim=1), torch.cat(masks, dim=1)
            for block in self.blocks:
                if use_condition:
                    q = block(q, qmask, c, cmask)
                else:
                    q = q + block.drop(block.self_attn(block.n1(q), block.n1(q), qmask))
                    q = q * qmask[:, :, None]
                    q = q + block.drop(block.ff(block.n3(q)))
                    q = q * qmask[:, :, None]
            state = self.final_norm(q[:, -1])
            if joint is not None and self.joint_enabled:
                role_mask = torch.tensor([use_condition, use_query, use_prior, True],
                                         device=device, dtype=torch.bool)
                if joint.shape[1] != 4:
                    raise ValueError('re-encoded A4 joint representation requires four named pool slots')
                joint_mask = (torch.ones(joint.shape[:2], device=device, dtype=torch.bool)
                              if joint_mask is None else joint_mask)
                joint_mask = joint_mask & role_mask[None, :]
                # Remove excluded slots before projection, so arbitrary NaN
                # payload in a removed slot cannot contaminate masked SDPA.
                retained = torch.nonzero(role_mask, as_tuple=True)[0]
                joint = self.joint_projection(joint[:, retained]) + self.joint_role
                joint_mask = joint_mask[:, retained]
                state = state + self.joint_attention(state[:, None, :], joint, joint_mask)[:, 0, :]
            if self.flat_output:
                logits = self.flat_head(state).float()
                # Exact three-way softmax represented by two log-odds; this
                # preserves CE and both ranking objectives in the base loop.
                changed = torch.logsumexp(logits[:, [0,2]], dim=1) - logits[:,1]
                direction = logits[:,0] - logits[:,2]
                return changed, direction
            return self.changed_head(state).squeeze(-1).float(), self.direction_head(state).squeeze(-1).float()

        head.forward = types.MethodType(ablated_forward, head)

    def parameter_report(self):
        return {'variant': self.branch_variant,
                'full_initialized_parameters': self.full_initialized_parameter_count,
                'retained_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'removed_parameters': self.full_initialized_parameter_count - sum(p.numel() for p in self.parameters()),
                'removed_branch': excluded,
                'joint_enabled': self.joint_enabled, 'flat_output': self.flat_output,
                'common_weights_initialized_in_full_order': True,
                'task_checkpoint_initialization': False,
                'old_prediction_output_anchor': False}

    head.parameter_report = types.MethodType(parameter_report, head)
    return head
