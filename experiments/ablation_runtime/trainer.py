#!/usr/bin/env python3
"""Isolated, fresh-head A4 branch ablations with the audited training loop.

This adapter replaces only input loading/model construction and provenance.
Optimization, exact exposure accounting, pair losses, validation selection,
convergence seal, independent restoration, and locked TEST evaluation are the
unchanged source train.py routines. No old task weights initialize any arm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import mmap
import os
from pathlib import Path
import sys
import types

import numpy as np

HERE = Path(__file__).resolve().parent
# Direct-script restore workers must also resolve the repository helpers.
REPOSITORY = HERE.parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / 'src')]
from experiments.ablation_runtime.contract import VARIANTS, roles
from jev_a4.core import train as packaged_training
from jev_a4.config import load_config
SOURCE = Path(packaged_training.__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
base = load_module('_a4_branch_audited_training', SOURCE / 'train.py')
from experiments.ablation_runtime import model as head_factory
original_model_config = base.model_config
original_settings = base.settings


def model_config(cfg):
    result = original_model_config(cfg)
    result['branch_variant'] = cfg.get('branch_variant', 'FULL')
    result['joint_enabled'] = cfg.get('joint_enabled', True)
    result['flat_output'] = cfg.get('flat_output', False)
    result['numeric_enabled'] = cfg.get('numeric_enabled', True)
    return result


def settings(cfg):
    result = original_settings(cfg)
    if cfg.get('smoke'):
        result['partition_ram_cache_budget_gib']=0
        result['preload_shared_gpu']=False
    if float(result['partition_ram_cache_budget_gib']) != 0 or result['preload_shared_gpu']:
        raise ValueError('ablation cache safety requires RAM preloading=0 and shared GPU preloading=false')
    return result


def variant_of(cfg):
    result = cfg.get('branch_variant', 'FULL')
    if result not in VARIANTS:
        raise ValueError('invalid complete branch ablation')
    return result


def release_mappings(arrays):
    """Release only this process's mmap PTEs; no global cache eviction."""
    for value in arrays:
        mapping = getattr(value, '_mmap', None)
        if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            try:
                mapping.madvise(mmap.MADV_DONTNEED)
            except OSError:
                pass


class SharedCache:
    def __init__(self, data_dir, cache_dir, cfg, device, guard=None):
        self.data_dir, self.cache_dir = Path(data_dir), Path(cache_dir)
        self.cfg, self.device = cfg, str(device)
        self.variant = variant_of(cfg)
        self.guard = guard or base.ResourceGuard(settings(cfg))
        self.base_cache_dir = Path(cfg.get('base_cache_dir', self.cache_dir))
        self.ram_cache_reports = []
        self.maps = []
        self.gpu_arrays = {}
        self.gpu_cache_bytes = 0
        if cfg.get('smoke'):
            self.base_cache_dir = Path(cfg.get('base_cache_dir', cache_dir))
        else:
            manifest = base.cache_metadata(self.base_cache_dir, 'shared_condition_query')
            if (manifest.get('response_labels_read') is not False or
                    manifest.get('old_task_weights_or_predictions_read') is not False):
                raise ValueError('standalone cache is not response-blind generic DECIDER')
        # Numeric array headers define axis lengths. Excluded payloads are not
        # mapped or read; the relevant counts come from generic cache headers.
        self.hidden = int(model_config(cfg)['backbone_hidden_width'])
        self.condition_count = self._shape(self.base_cache_dir / 'condition_final32.npy')[0]
        self.query_count = self._shape(self.base_cache_dir / 'query_final32.npy')[0]
        self.condition = self.condition_num = self.query = self.query_num = None
        if roles(self.variant)[0]:
            self.condition = self.load(self.base_cache_dir / 'condition_final32.npy')
            self.condition_num = self.load(self.data_dir / 'condition_numeric.npy')
            if self.condition.shape != (self.condition_count, 16, self.hidden) or self.condition_num.shape != (self.condition_count, 4):
                raise ValueError('condition semantic/numeric axis mismatch')
        if roles(self.variant)[1]:
            self.query = self.load(self.base_cache_dir / 'query_final32.npy')
            self.query_num = self.load(self.data_dir / 'query_numeric.npy')
            if self.query.shape != (self.query_count, 16, self.hidden) or self.query_num.shape != (self.query_count, 4):
                raise ValueError('query semantic/numeric axis mismatch')
        for array in (self.condition,self.query):
            if array is not None:
                self.stage_gpu(array)

    def stage_gpu(self, array):
        import torch
        if not self.device.startswith('cuda') or id(array) in self.gpu_arrays:
            return
        free,_ = torch.cuda.mem_get_info()
        budget = min(float(self.cfg.get('gpu_cache_gib',48))*2**30-self.gpu_cache_bytes, free-20*2**30)
        if array.nbytes > budget:
            return
        gpu = torch.empty(array.shape,dtype=torch.float16,device=self.device)
        step=max(1,(32<<20)//int(np.prod(array.shape[1:])*array.dtype.itemsize))
        for start in range(0,len(array),step):
            host=np.array(array[start:start+step],copy=True)
            if not np.isfinite(host).all():
                raise FloatingPointError('nonfinite cached representation')
            gpu[start:start+len(host)].copy_(torch.from_numpy(host))
            # Verify the dtype-preserving placement on both boundaries.
            if start == 0 or start+step >= len(array):
                if not np.array_equal(gpu[start:start+len(host)].cpu().numpy(),host):
                    raise ValueError('GPU cache placement changed values')
            release_mappings([array])
        self.gpu_arrays[id(array)]=gpu
        self.gpu_cache_bytes += array.nbytes

    def cached_gather(self,array,indices):
        import torch
        if id(array) in self.gpu_arrays:
            idx=torch.as_tensor(np.array(indices,copy=True),device=self.device,dtype=torch.long)
            return self.gpu_arrays[id(array)][idx].to(torch.bfloat16)
        return self.tensor(array[indices],floating=True)

    @staticmethod
    def _shape(path):
        with Path(path).open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            shape, _, _ = np.lib.format._read_array_header(stream, version)
        return shape

    def load(self, path):
        result = np.load(path, mmap_mode='r', allow_pickle=False)
        self.maps.append(result)
        return result

    def tensor(self, array, floating=False):
        import torch
        result = torch.from_numpy(np.array(array, copy=True, order='C'))
        if self.device.startswith('cuda'):
            result = result.pin_memory().to(self.device, non_blocking=True)
            if floating:
                result = result.to(torch.bfloat16)
        elif floating:
            result = result.float()
        return result

    def gather(self, field, indices):
        return self.cached_gather(getattr(self, field), indices)


def _leaf_contract(shared, seed, name):
    cache = shared.cache_dir / f'seed_{seed}' / name
    if shared.cfg.get('smoke'):
        return cache
    leaf = json.loads((cache / 'manifest.json').read_text())
    if leaf.get('schema') == 'decider.a4.paper.encoding.v1':
        if leaf.get('status') != 'PASS':
            raise ValueError('original cache not sealed PASS')
        inputs = leaf.get('identity', {}).get('inputs', {})
        for filename in ('records.npy','prior_features.npy'):
            if base.sha256(shared.data_dir / f'seed_{seed}' / name / filename) != inputs.get(filename):
                raise ValueError('source cache record/evidence binding mismatch')
        return cache
    if leaf.get('status') != 'PASS' or leaf.get('variant') != shared.variant:
        raise ValueError('joint cache is not a sealed matching branch ablation')
    if leaf.get('schema') != 'decider.a4.branch_ablation.encoding.v1':
        raise ValueError('unexpected complete branch cache schema')
    if leaf.get('labels_read') is not False or leaf.get('task_weights_or_predictions_read') is not False:
        raise ValueError('joint encoder read labels or task outputs')
    if leaf.get('frozen_last8') is not True:
        raise ValueError('joint input must be freshly compiled through frozen last8')
    expected_mask = [*roles(shared.variant), True]
    if leaf.get('joint_pool_mask') != expected_mask:
        raise ValueError('removed branch retained in named joint pools')
    inputs = leaf.get('inputs', leaf.get('identity', {}).get('inputs', {}))
    data = shared.data_dir / f'seed_{seed}' / name
    for filename in ('records.npy', 'prior_features.npy'):
        if base.sha256(data / filename) != inputs.get(filename):
            raise ValueError('joint representation and source record/evidence identity differ')
    payload = Path(leaf.get('payload_directory', leaf.get('source_directory', str(cache))))
    if not payload.is_absolute():
        payload = cache / payload
    return payload


class Partition:
    def __init__(self, shared, seed, name):
        if name not in ('train', 'val', 'test'):
            raise ValueError('unknown data partition')
        self.shared, self.name = shared, name
        data = shared.data_dir / f'seed_{seed}' / name
        cache = _leaf_contract(shared, seed, name)
        self.records = shared.load(data / 'records.npy')
        # Read-only features serve shape/provenance for every arm and the
        # independent P1 comparator at TEST; NO_PRIOR.batch never reads values.
        self.features = shared.load(data / 'prior_features.npy')
        self.joint_present = self.joint_absent = self.joint_ids = None
        if shared.cfg.get('joint_enabled',True):
            self.joint_present = shared.load(cache / 'joint_present.npy')
            self.joint_absent = shared.load(cache / 'joint_absent.npy')
            self.joint_ids = shared.load(cache / 'joint_ids.npy')
            shared.stage_gpu(self.joint_present)
            if not roles(shared.variant)[2]:
                if id(self.joint_present) in shared.gpu_arrays:
                    shared.gpu_arrays[id(self.joint_absent)] = shared.gpu_arrays[id(self.joint_present)]
            else:
                shared.stage_gpu(self.joint_absent)
        self.cards = self.card_ids = None
        if roles(shared.variant)[2]:
            self.cards = shared.load(shared.base_cache_dir / f'seed_{seed}' / 'cards_final.npy')
            self.card_ids = shared.load(shared.base_cache_dir / f'seed_{seed}' / name / 'card_ids.npy')
            if self.cards.shape[1:] != (4, shared.hidden):
                raise ValueError('prior semantic representation axis mismatch')
            if self.card_ids.shape != (len(self.records),) or np.min(self.card_ids) < 0 or np.max(self.card_ids) >= len(self.cards):
                raise ValueError('invalid prior card ids')
        if not self.records.dtype.names or not {'target', 'query', 'label'} <= set(self.records.dtype.names):
            raise ValueError('invalid record schema')
        if not len(self.records) or self.features.shape != (len(self.records), int(shared.cfg.get('prior_feature_width', 12))):
            raise ValueError('empty records or evidence width drift')
        if shared.cfg.get('joint_enabled',True) and (self.joint_present.shape[1:] != (4, shared.hidden) or self.joint_absent.shape != self.joint_present.shape or
                self.joint_ids.shape != (len(self.records),)):
            raise ValueError('record/joint axes mismatch')
        if self.joint_ids is not None and (np.min(self.joint_ids) < 0 or np.max(self.joint_ids) >= len(self.joint_present)):
            raise ValueError('invalid joint tuple ids')
        if np.any((self.records['label'] < 0) | (self.records['label'] > 2)):
            raise ValueError('non-strict labels entered task training')
        if np.any((self.records['target'] < 0) | (self.records['target'] >= shared.condition_count)):
            raise ValueError('invalid target axis')
        if np.any((self.records['query'] < 0) | (self.records['query'] >= shared.query_count)):
            raise ValueError('invalid query axis')
        shared.ram_cache_reports.append({'partition': name, 'mode': 'READ_ONLY_MEMMAP_NO_PRELOAD',
                                         'self_mmap_PTE_release_each_batch': True,
                                         'removed_input_branch': shared.variant})
        shared.ram_cache_reports[-1].update(mode='BOUNDED_HOST_GPU_CACHE',
            gpu_cache_gib=shared.gpu_cache_bytes/2**30, host_staging_mib=32)
        release_mappings(shared.maps)

    def batch(self, indices, keep=None):
        import torch
        indices = np.asarray(indices, dtype=np.int64)
        keep = np.ones(len(indices), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
        if self.shared.cfg.get('forced_missing_prior'):
            keep = np.zeros(len(indices),dtype=bool)
        if keep.shape != indices.shape:
            raise ValueError('one prior mask must bind all evidence channels')
        s = self.shared
        rows = self.records[indices]
        n, h, device = len(rows), s.hidden, s.device
        def empty(length, width):
            return torch.empty((length, width, h), device=device,
                               dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32)
        if roles(s.variant)[0]:
            unique, inverse = np.unique(rows['target'], return_inverse=True)
            condition = s.gather('condition', unique)
            condition_num = s.tensor(s.condition_num[unique]).float()
            condition_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
        else:
            condition = empty(1, 0)
            condition_num = torch.empty((1, 0), device=device)
            condition_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
            inverse = np.zeros(n, dtype=np.int64)
        if roles(s.variant)[1]:
            query = s.gather('query', np.asarray(rows['query'], dtype=np.int64))
            query_num = s.tensor(s.query_num[rows['query']]).float()
            query_mask = torch.ones(query.shape[:2], dtype=torch.bool, device=device)
        else:
            query = empty(n, 0)
            query_num = torch.empty((n, 0), device=device)
            query_mask = torch.zeros((n, 0), dtype=torch.bool, device=device)
        if roles(s.variant)[2]:
            prior = s.tensor(self.cards[self.card_ids[indices]], floating=True)
            numeric = np.array(self.features[indices], dtype=np.float32, copy=True)
            numeric[~keep] = 0
            numeric[~keep, :3] = 1 / 3
            numeric[~keep, -1] = 0
            if not np.isfinite(numeric).all():
                raise FloatingPointError('non-finite prior features')
            prior_num = s.tensor(numeric).float()
            prior_mask = torch.as_tensor(keep[:, None].repeat(4, axis=1), device=device)
            prior[~torch.as_tensor(keep,device=device)] = 0
        else:
            prior = empty(n, 0)
            prior_num = torch.empty((n, 0), device=device)
            prior_mask = torch.zeros((n, 0), dtype=torch.bool, device=device)
            keep = np.zeros(n, dtype=bool)
        joint = None
        if s.cfg.get('joint_enabled',True):
            ids = self.joint_ids[indices]
            joint = s.cached_gather(self.joint_present,ids)
            if not keep.all() and not (s.cfg.get('unsynchronized_masking') and self.name == 'train'):
                dropped=torch.as_tensor(~keep,device=device)
                joint[dropped] = s.cached_gather(self.joint_absent,ids[~keep])
            if not torch.isfinite(joint).all():
                raise FloatingPointError('non-finite re-encoded joint states')
        mask = np.array([*roles(s.variant), True], dtype=bool)
        joint_mask = torch.as_tensor(np.broadcast_to(mask, (n, 4)).copy(), device=device)
        # Preserve the original A4 dropout contract: the joint absent-cache
        # prior slot is a neutral zero pool, and its learned joint role remains
        # visible. Only a globally removed branch is physically excluded.
        result = (condition, condition_mask, condition_num, query, query_mask, query_num,
                  prior, prior_mask, prior_num, torch.as_tensor(inverse, device=device, dtype=torch.long),
                  joint, joint_mask, None)
        if s.cfg.get('numeric_only'):
            # Exact old NO_JEV intervention: numeric tokens retained, all
            # semantic landmarks zero/masked and no tuple-level joint.
            result=list(result)
            for a,b in ((0,1),(3,4),(6,7)):
                result[a]=torch.zeros_like(result[a]);result[b]=torch.zeros_like(result[b])
            result[10]=None
            result=tuple(result)
        release_mappings(s.maps)
        return result


def bindings(cfg, data_dir, cache_dir, seed):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    manifests = {}
    paths = [data_dir / 'dataset.json', data_dir / 'preparation_report.json',
             data_dir / f'seed_{seed}' / 'manifest.json',
             data_dir / f'seed_{seed}' / 'target_partition.npy',
             data_dir / f'seed_{seed}' / 'target_fold.npy',
             data_dir / f'seed_{seed}' / 'split_manifest.json',
             data_dir / f'seed_{seed}' / 'audit.json',
             Path(cfg.get('base_cache_dir', cache_dir)) / 'manifest.json']
    for name in ('train', 'val', 'test'):
        paths.append(cache_dir / f'seed_{seed}' / name / 'manifest.json')
    for path in paths:
        if path.is_file():
            manifests[str(path)] = base.sha256(path)
    return {'config_sha256': base.object_sha(cfg), 'manifest_sha256': manifests,
            'contract_sha256': base.sha256(HERE / 'contract.py'),
            'trainer_sha256': base.sha256(HERE / 'trainer.py'),
            'branch_model_sha256': base.sha256(HERE / 'model.py'),
            'audited_training_loop_sha256': base.sha256(SOURCE / 'train.py'),
            'generic_factory_sha256': base.sha256(SOURCE / 'vendor' / 'prior_model.py'),
            'metrics_sha256': base.sha256(SOURCE / 'metrics.py'),
            'variant': variant_of(cfg)}


def selected_checkpoint(model, probabilities, metrics, epoch, cfg, seed, dataset, out, bound):
    meta = {'dataset': dataset, 'seed': int(seed), 'selected_epoch': int(epoch),
            'model_config': model_config(cfg), 'prior_feature_width': model.prior_feature_width,
            'bindings': bound, 'validation_metrics': metrics, 'branch_variant': model.branch_variant,
            'parameters': model.parameter_report(), 'fresh_task_head': True,
            'old_task_checkpoint_initialization': False, 'direct_three_class_prediction': True,
            'lora': False, 'direct_relations': False, 'old_prediction_output_anchor': False,
            'output_residual_correction': False, 'parameter_update_labels': ['TRAIN'], 'selection_labels': ['VAL']}
    base.save_checkpoint(out / 'best.pt', model, meta)
    base.atomic_npz(out / 'best_val_predictions.npz', probabilities=probabilities,
                    checkpoint_sha256=np.asarray(base.sha256(out / 'best.pt')))
    base.atomic_json(out / 'model_parameters.json', model.parameter_report())


base.CHECKPOINT_SCHEMA = 'decider.a4.complete_branch_ablation.fresh_head.v1'
base.SharedCache = SharedCache
base.Partition = Partition
base.model_config = model_config
base.settings = settings
base.vendor_module = lambda: types.SimpleNamespace(make_prior_model=head_factory.make_prior_model)
base.bindings = bindings
base.selected_checkpoint = selected_checkpoint
# The source loop starts an independent restore worker using its __file__.
# Point that launch at this adapter; source code itself remains unchanged.
base.RESTORE_ENTRYPOINT = HERE / 'trainer.py'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'evaluate', 'restore-check'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    variant_of(cfg)
    settings(cfg)
    import torch
    if cfg.get('device', 'cuda').startswith('cuda') and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        ceiling = float(settings(cfg)['gpu_memory_limit_gib']) * 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(ceiling / total, .95), 0)
    try:
        action = {'train': base.train, 'evaluate': base.evaluate_test, 'restore-check': base.restore_check}[args.action]
        result = action(cfg, args.dataset, args.seed, args.data_dir, args.cache_dir, args.out)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        base.atomic_json(args.out / f'{args.action}_failure.json',
                         {'status': 'FAIL', 'error_type': type(error).__name__, 'error': str(error),
                          'dataset': args.dataset, 'seed': args.seed, 'variant': variant_of(cfg),
                          'automatic_restart': False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
