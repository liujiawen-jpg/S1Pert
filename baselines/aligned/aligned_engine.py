"""A4 adapter around the pinned official ALIGNED MLP and RegulatoryKB.

The runner owns splits, the two-round schedule, selection and atomic checkpoints.
An iteration always finishes one optimizer update before a checkpoint is valid.
No held-out labels are accepted by the engine: pass TRAIN tensors only.

Disclosed execution fixes: mask unresolved CE entries; reuse one permutation for
both REINFORCE passes; restore train mode after inference. We preserve the official
probabilities-to-CrossEntropyLoss behavior, batch-mean CE summed across batches,
global reward, per-batch EMA, sparse_opt_tanh's nonnegative clamp and early stop.
"""
import copy
import inspect
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

OFFICIAL_SOURCE = Path(__file__).resolve().parent / 'official_source'
sys.path.insert(0, str(OFFICIAL_SOURCE))
from aligned.learner_adap import AdaptorMLP, AdaptorLearner
from aligned.reasoner import RegulatoryKB, tanh_power, tanh_soft

for component in (AdaptorMLP, RegulatoryKB):
    if OFFICIAL_SOURCE not in Path(inspect.getfile(component)).resolve().parents:
        raise RuntimeError('ALIGNED must use the pinned official source tree')

DEFAULTS = dict(hidden_dim=64, batch_size=64, rounds=2,
    weight_init_epochs=1000, weight_init_lr=1e-4,
    pretrain_epochs=300, pretrain_reinforce_epochs=1, pretrain_C=10.,
    retrain_epochs=150, retrain_reinforce_epochs=100, retrain_C=100.,
    training_lr=1e-3, refine_epochs=5000, refine_lr=1e-3,
    closure=5, closure_type='weighted', refine_C=1., refine_t=1.,
    refine_t0=100., refine_tol=1e-3, gamma=.95)
ADAPTATIONS = {
    'core': 'official AdaptorMLP + official RegulatoryKB + official consistency_reward',
    'ce': 'official softmax probabilities passed to CrossEntropyLoss; unknown entries excluded',
    'class_weights': 'initial [.4,.2,.4]; inverse-frequency only on update_weight=True',
    'reinforce': 'single fixed permutation shared by reward and gradient passes',
    'backward': 'per-batch gradient accumulation; one Adam update per full iteration',
    'disabled_ce': 'skip coefficient-zero CE evaluation but retain explicit zero gradients and Adam moment decay',
    'weight_init': 'identity inputs chunked; only r_head trained, unchanged full MSE objective',
    'refine': 'official sparse_opt_tanh objective/Adam/clamp(min=0)/rounding/tol preserved',
    'output_axis': 'all genes, identity output_idx_list; no double-subset indexing',
    'training_mode': 'model.train restored at each optimizer iteration',
}
KB_TENSORS = ('Regu_P_0', 'Regu_N_0', 'Regu_0', 'KB_P', 'KB_N', 'KB')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG checkpoint requires a CUDA runtime')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def class_weights(labels, known_mask=None):
    """Official inverse counts, restricted to observed TRAIN entries."""
    known = labels != -100 if known_mask is None else known_mask.bool()
    values = labels[known]
    if torch.any((values < -1) | (values > 1)):
        raise ValueError('known labels must be -1, 0, +1')
    # Match official Python float inverse counts followed by float32 conversion.
    weights = [1 / (int((values == label).sum()) + 1e-6) for label in (-1, 0, 1)]
    return torch.tensor(weights, dtype=torch.float32, device=labels.device) / sum(weights)


def masked_ce(probabilities, labels, known_mask, weights):
    """Official weighted mean CE, with an attached zero for empty minibatches."""
    known = known_mask.bool()
    if not bool(known.any()):
        return probabilities.sum() * 0.
    selected = labels[known]
    if torch.any((selected < -1) | (selected > 1)):
        raise ValueError('known labels must be -1, 0, +1')
    return nn.functional.cross_entropy(probabilities[known], selected.long() + 1,
                                       weight=weights)


def _check_identity_axis(kb, size):
    if list(kb.idx_list) != list(range(size)) or tuple(kb.KB.shape) != (size, size):
        raise ValueError('RegulatoryKB must use the complete shared gene axis and identity output indices')


class TanhRefiner:
    """Interruptible transcription of RegulatoryKB.sparse_opt_tanh.

    Calls the official tanh_power/tanh_soft functions. All optimizer state is
    persistent; stopping and restoring never reinitializes Adam or its anchor X0.
    """
    def __init__(self, kb, X, Y, *, k=5, epochs=5000, lr=1e-3,
                 C=1., t=1., t0=100., tol=1e-3):
        _check_identity_axis(kb, kb.Regu_0.shape[0])
        if epochs < 1 or k < 1:
            raise ValueError('refine epochs and matrix power must be positive')
        self.kb = kb
        self.config = dict(k=k, epochs=epochs, lr=lr, C=C, t=t, t0=t0, tol=tol)
        device = kb.Regu_0.device
        X = torch.as_tensor(X, dtype=kb.Regu_0.dtype, device=device)
        Y = torch.as_tensor(Y, device=device)
        if (X.ndim != 2 or len(X) == 0 or X.shape[1] != kb.Regu_0.shape[0]
                or Y.shape != (len(X), kb.Regu_0.shape[0])
                or not torch.all((Y == -1) | (Y == 0) | (Y == 1))):
            raise ValueError('refinement requires all TRAIN targets and fully resolved pseudo/KB labels')
        self.anchor = kb.Regu_0.detach().clone()
        self.data = (X.T @ Y.float()).clamp(-1, 1)
        self.omega = torch.any(self.data != 0, axis=1)
        self.matrix = self.anchor.clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.matrix], lr=lr)
        self.losses = []
        self.converged = False
        self.finalized = False

    @property
    def done(self):
        return self.converged or len(self.losses) >= self.config['epochs']

    def step(self):
        if self.done:
            raise RuntimeError('refine is already complete')
        self.optimizer.zero_grad()
        cfg = self.config
        powered = tanh_power(self.matrix, cfg['k'], cfg['t'])
        data_loss = torch.norm((powered[:, self.kb.idx_list] - self.data)[self.omega], p='fro') ** 2
        regularization = torch.norm(tanh_soft(self.matrix, cfg['t0']) - self.anchor, p=1)
        loss = data_loss + cfg['C'] * regularization
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite official refine objective')
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.matrix.copy_(self.matrix.clamp(min=0))
        self.losses.append(loss.item())
        self.converged = len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < cfg['tol']
        return {'mode': 'refine', 'iteration': len(self.losses), 'step': len(self.losses), 'total_iterations': cfg['epochs'],
                'loss': loss.item(), 'data_loss': data_loss.item(),
                'regularization': regularization.item(), 'done': self.done,
                'converged': self.converged}

    @torch.no_grad()
    def finalize(self):
        if not self.done:
            raise RuntimeError('refine must finish its configured budget or official tolerance first')
        matrix, cfg = self.matrix.detach(), self.config
        self.kb.KB = torch.round(tanh_power(matrix, cfg['k'], cfg['t'])).clamp(-1, 1)
        self.kb.KB_P, self.kb.KB_N = self.kb.KB.clamp(0, 1), (-self.kb.KB).clamp(0, 1)
        self.kb.Regu_0 = matrix.round().clamp(-1, 1)
        self.kb.Regu_P_0 = matrix.round().clamp(0, 1)
        self.kb.Regu_N_0 = (-matrix).round().clamp(0, 1)
        self.finalized = True

    def state_dict(self):
        return cpu_tree(dict(config=self.config, anchor=self.anchor, data=self.data,
            omega=self.omega, matrix=self.matrix, optimizer=self.optimizer.state_dict(),
            losses=self.losses, converged=self.converged, finalized=self.finalized))

    @classmethod
    def from_state_dict(cls, kb, state):
        result = cls.__new__(cls)
        result.kb = kb
        _check_identity_axis(kb, kb.Regu_0.shape[0])
        result.config = copy.deepcopy(state['config'])
        device = kb.Regu_0.device
        for key in ('anchor', 'data', 'omega'):
            setattr(result, key, state[key].to(device).clone())
        result.matrix = state['matrix'].to(device).clone().requires_grad_(True)
        result.optimizer = torch.optim.Adam([result.matrix], lr=result.config['lr'])
        result.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        result.losses = list(state['losses'])
        result.converged, result.finalized = state['converged'], state['finalized']
        return result


class AlignedEngine:
    def __init__(self, input_dim, output_dim, kb, label_weight, *, device='cpu',
                 batch_size=64, hidden_dim=64):
        if input_dim != output_dim or batch_size < 1:
            raise ValueError('A4 requires one complete shared gene axis and a positive batch size')
        _check_identity_axis(kb, input_dim)
        self.device = torch.device(device)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim, self.batch_size = hidden_dim, batch_size
        self.kb = kb
        self.kb.device = str(self.device)
        for name in KB_TENSORS:
            setattr(self.kb, name, getattr(self.kb, name).to(self.device))
        self.discretized = True
        self.model = AdaptorMLP(input_dim, hidden_dim, output_dim, discretized=True).to(self.device)
        self.label_weight = torch.as_tensor(label_weight, dtype=torch.float32, device=self.device)
        if self.label_weight.shape != (output_dim,) or not torch.isfinite(self.label_weight).all():
            raise ValueError('label_weight must be a finite vector on the full gene axis')
        if torch.any((self.label_weight < 0) | (self.label_weight > 1)):
            raise ValueError('label_weight must be in [0,1]')
        self.clf_weight = torch.tensor([.4, .2, .4], device=self.device)
        self.X = self.Y = self.known_mask = None
        self.optimizer = None
        self.refiner = None
        self.mode, self.stage = None, None
        self.config = {}
        self.iteration, self.baseline = 0, 0.
        self.early_stop = False

    @property
    def KB(self):
        return self.kb

    @property
    def done(self):
        if self.mode == 'refine':
            return self.refiner.done
        return self.early_stop or self.iteration >= self.config.get('total_iterations', 0)

    def set_data(self, X, Y, known_mask=None, *, update_weight=False):
        X = torch.as_tensor(X, dtype=torch.float32).detach().cpu().clone()
        Y = torch.as_tensor(Y).detach().cpu().clone()
        if X.ndim != 2 or len(X) == 0 or X.shape[1] != self.input_dim or Y.shape != (len(X), self.output_dim):
            raise ValueError('TRAIN matrices do not match the engine axes')
        known = Y != -100 if known_mask is None else torch.as_tensor(known_mask).detach().cpu().bool().clone()
        if known.shape != Y.shape or torch.any(known & ((Y < -1) | (Y > 1))):
            raise ValueError('known TRAIN labels must be -1, 0, +1')
        self.X, self.Y, self.known_mask = X, Y.long(), known
        if update_weight:
            self.clf_weight = class_weights(self.Y, known).to(self.device)

    def _begin(self, mode, stage, config, *, r_only=False):
        self.mode, self.stage, self.config = mode, stage, config
        self.iteration, self.baseline, self.early_stop = 0, 0., False
        self.refiner = None
        for name, param in self.model.named_parameters():
            param.requires_grad_(not r_only or name.startswith('r_head'))
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])

    def begin_training(self, stage, *, epochs, reinforce_epochs, C, lr=1e-3, gamma=.95):
        if self.X is None or epochs < 1 or reinforce_epochs < 1:
            raise ValueError('set TRAIN data first and use positive iteration counts')
        self._begin('training', stage, dict(epochs=epochs, reinforce_epochs=reinforce_epochs,
            C=C, lr=lr, gamma=gamma, total_iterations=epochs * reinforce_epochs))

    def train_iteration(self, *, batch_callback=None, monitor_disabled_ce=False):
        if self.mode != 'training' or self.done:
            raise RuntimeError('no unfinished training stage')
        self.model.train()
        order = torch.randperm(len(self.X))
        batches = list(order.split(self.batch_size))
        actions, reward = [], 0.
        with torch.no_grad():
            for rows in batches:
                x = self.X[rows].to(self.device)
                output_y, output_r = self.model(x)
                sampled = Bernoulli(output_r.clamp(.2, .8)).sample()
                actions.append(sampled.detach())
                reward += AdaptorLearner.consistency_reward(self, self.kb, x, output_y,
                    sampled, self.label_weight).item()
                if batch_callback:
                    batch_callback({'phase': 'reward', 'rows': rows.tolist(), 'actions': sampled.detach().cpu().clone()})
        self.optimizer.zero_grad()
        ce_enabled = self.iteration % self.config['reinforce_epochs'] == 0
        ce_sum, rl_sum = 0., 0.
        for rows, sampled in zip(batches, actions):
            x = self.X[rows].to(self.device)
            output_y, output_r = self.model(x)
            if ce_enabled or monitor_disabled_ce:
                ce = masked_ce(output_y, self.Y[rows].to(self.device),
                               self.known_mask[rows].to(self.device), self.clf_weight)
            else:
                # An explicit attached zero keeps Adam's updates to previous
                # classifier moments identical; grad=None would change them.
                ce = output_y.sum() * 0.
            gamma = self.config['gamma']
            self.baseline = gamma * self.baseline + (1 - gamma) * reward
            log_probs = Bernoulli(output_r).log_prob(sampled).sum(dim=1)
            rl = -torch.mean((reward - self.baseline) * log_probs)
            loss = int(ce_enabled) * ce + self.config['C'] * rl
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite ALIGNED training objective')
            # Linear sum over independent batch graphs; parameters change only
            # after the complete second pass, exactly one update per iteration.
            loss.backward()
            ce_sum += ce.item()
            rl_sum += rl.item()
            if batch_callback:
                batch_callback({'phase': 'gradient', 'rows': rows.tolist(), 'actions': sampled.detach().cpu().clone()})
        self.optimizer.step()
        self.iteration += 1
        return {'mode': self.mode, 'stage': self.stage, 'iteration': self.iteration, 'step': self.iteration,
            'total_iterations': self.config['total_iterations'], 'done': self.done,
            'ce_loss': ce_sum, 'rl_loss': rl_sum,
            'total_loss': int(ce_enabled) * ce_sum + self.config['C'] * rl_sum,
            'ce_enabled': ce_enabled, 'ce_evaluated': ce_enabled or monitor_disabled_ce,
            'reward': reward, 'baseline': self.baseline,
            'optimizer_updates': self.iteration, 'train_targets': len(self.X),
            'known_labels': int(self.known_mask.sum())}

    def begin_weight_init(self, *, epochs=1000, lr=1e-4):
        if epochs < 1:
            raise ValueError('weight initialization epochs must be positive')
        self._begin('weight_init', 'weight_init', dict(lr=lr, total_iterations=epochs), r_only=True)

    def weight_init_iteration(self):
        if self.mode != 'weight_init' or self.done:
            raise RuntimeError('no unfinished weight initialization')
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.
        for start in range(0, self.input_dim, self.batch_size):
            end = min(start + self.batch_size, self.input_dim)
            x = torch.zeros((end - start, self.input_dim), device=self.device)
            x[torch.arange(end - start, device=self.device), torch.arange(start, end, device=self.device)] = 1
            # The unused classification head has no contribution to the original
            # weight-init MSE; omit that branch without changing R or gradients.
            emb = self.model.embedding(x)
            r = self.model.sigmoid(self.model.r_head(emb))
            loss = nn.functional.mse_loss(r, self.label_weight.expand_as(r), reduction='sum') / (self.input_dim * self.output_dim)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()
        self.iteration += 1
        self.early_stop = total_loss < 1e-4
        if self.done:
            self.model.requires_grad_(True)
        return {'mode': self.mode, 'stage': self.stage, 'iteration': self.iteration, 'step': self.iteration,
            'total_iterations': self.config['total_iterations'], 'loss': total_loss,
            'done': self.done, 'early_stop': self.early_stop}

    @torch.no_grad()
    def predict(self, X, batch_size=64, *, include_r=True):
        """Fixed hybrid main prediction. Accepts features only, never VAL/TEST labels.

        Returns CPU tensors hybrid/neural/deduction [rows, genes] in {-1,0,1}
        and optionally R [rows, genes]. No metric-based choice between outputs.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] != self.input_dim or batch_size < 1:
            raise ValueError('nonempty feature matrix on the full gene axis required')
        self.model.eval()
        parts = {key: [] for key in ('hybrid', 'neural', 'deduction')}
        if include_r:
            parts['R'] = []
        for x in X.split(batch_size):
            x = x.to(self.device)
            probs, r = self.model(x)
            pseudo = probs.argmax(-1) - 1
            deduction = self.kb.deduce(x).long()
            modified = torch.where(r > .5, deduction, pseudo)
            for key, value in (('hybrid', modified), ('neural', pseudo), ('deduction', deduction)):
                parts[key].append(value.cpu())
            if include_r:
                parts['R'].append(r.cpu())
        return {key: torch.cat(value) for key, value in parts.items()}

    def pseudo_labels(self, X=None):
        X = self.X if X is None else X
        if X is None:
            raise ValueError('nonempty TRAIN targets required')
        result = self.predict(X, batch_size=self.batch_size)
        return {'modified': result['hybrid'], 'pseudo': result['neural'],
                'R': result['R'], 'deduction': result['deduction']}

    def begin_refine(self, X, Y, *, stage='refine', **kwargs):
        self.mode, self.stage = 'refine', stage
        self.optimizer = None
        self.refiner = TanhRefiner(self.kb, X, Y, **kwargs)
        self.iteration = 0

    def refine_iteration(self):
        if self.mode != 'refine':
            raise RuntimeError('no refine stage')
        result = self.refiner.step()
        self.iteration = result['iteration']
        result['stage'] = self.stage
        if self.refiner.done:
            self.refiner.finalize()
        return result

    def run(self, callback=None, *, max_steps=None):
        """Run remaining iterations; callback(progress) may atomically save state."""
        count, latest = 0, None
        while not self.done and (max_steps is None or count < max_steps):
            step = {'training': self.train_iteration, 'weight_init': self.weight_init_iteration,
                    'refine': self.refine_iteration}[self.mode]
            latest = step()
            count += 1
            if callback:
                callback(latest)
        return latest

    def state_dict(self):
        return cpu_tree({'version': 1, 'axes': (self.input_dim, self.output_dim, self.hidden_dim),
            'batch_size': self.batch_size, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'kb': {name: getattr(self.kb, name) for name in KB_TENSORS},
            'kb_T': getattr(self.kb, 'T', None), 'label_weight': self.label_weight,
            'clf_weight': self.clf_weight, 'X': self.X, 'Y': self.Y, 'known_mask': self.known_mask,
            'mode': self.mode, 'stage': self.stage, 'config': self.config,
            'iteration': self.iteration, 'baseline': self.baseline, 'early_stop': self.early_stop,
            'refiner': self.refiner.state_dict() if self.refiner else None,
            'rng': rng_state(), 'adaptations': ADAPTATIONS})

    def load_state_dict(self, state):
        if state['version'] != 1 or tuple(state['axes']) != (self.input_dim, self.output_dim, self.hidden_dim):
            raise ValueError('incompatible ALIGNED checkpoint axes/version')
        if state['batch_size'] != self.batch_size or state['adaptations'] != ADAPTATIONS:
            raise ValueError('ALIGNED execution configuration changed')
        self.model.load_state_dict(state['model'])
        for name in KB_TENSORS:
            setattr(self.kb, name, state['kb'][name].to(self.device).clone())
        if state['kb_T'] is not None:
            self.kb.T = state['kb_T']
        self.label_weight = state['label_weight'].to(self.device).clone()
        self.clf_weight = state['clf_weight'].to(self.device).clone()
        for name in ('X', 'Y', 'known_mask'):
            setattr(self, name, cpu_tree(state[name]))
        for name in ('mode', 'stage', 'config', 'iteration', 'baseline', 'early_stop'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.refiner = TanhRefiner.from_state_dict(self.kb, state['refiner']) if state['refiner'] else None
        self.optimizer = None
        if state['optimizer'] is not None:
            r_only = self.mode == 'weight_init'
            for name, param in self.model.named_parameters():
                param.requires_grad_(not r_only or name.startswith('r_head'))
            self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.config['lr'])
            self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            if r_only and self.done:
                self.model.requires_grad_(True)
        restore_rng(state['rng'])
