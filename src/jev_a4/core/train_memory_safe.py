"""Memory-bounded A4 execution: unchanged scientific model, loss and schedule.

All production entry points install this layer before invoking ``train``.
Independent checkpoint replay uses this same file in a fresh interpreter.
"""
import mmap
from pathlib import Path
import numpy as np

from jev_a4.core import train

_INSTALLED = False


def release_mapping(array):
    mapping = getattr(array, '_mmap', None)
    advice = getattr(mmap, 'MADV_DONTNEED', None)
    if mapping is not None and hasattr(mapping, 'madvise') and advice is not None:
        try:
            mapping.madvise(advice)
        except (OSError, ValueError):
            # This is an optional OS cache hint, never a data mutation.
            pass


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    original_batch = train.Partition.batch
    original_bindings = train.bindings
    overlay_sha = train.sha256(Path(__file__))

    def preload(shared, partition):
        import torch
        # CPU whole-partition preload is forbidden by this execution overlay.
        # GPU caches preserve the stored dtype; conversion to BF16 stays at gather.
        required = int(partition.joint_present.nbytes + partition.joint_absent.nbytes)
        used = getattr(shared, '_joint_gpu_used', 0)
        partition._gpu_joints = None
        partition._parity_checked = set()
        if str(shared.device).startswith('cuda'):
            free, _ = torch.cuda.mem_get_info(shared.device)
            budget = min(32 * 2**30 - used, max(0, free - 14 * 2**30))
            if required <= budget:
                staged = []
                for array in (partition.joint_present, partition.joint_absent):
                    dtype = torch.from_numpy(np.array(array[:1], copy=True)).dtype
                    gpu = torch.empty(array.shape, device=shared.device, dtype=dtype)
                    row_bytes = int(np.prod(array.shape[1:]) * array.dtype.itemsize)
                    chunk = max(1, (64 << 20) // row_bytes)
                    for start in range(0, len(array), chunk):
                        host = np.array(array[start:start + chunk], copy=True)
                        if not np.isfinite(host).all():
                            raise FloatingPointError('Non-finite sealed cache during GPU preload')
                        gpu[start:start + len(host)].copy_(torch.from_numpy(host))
                        del host
                        release_mapping(array)
                    staged.append(gpu)
                partition._gpu_joints = tuple(staged)
                shared._joint_gpu_used = used + required
        mode = 'GPU_CHUNKED_PRELOAD' if partition._gpu_joints is not None else 'MEMMAP_BOUNDED_GATHER'
        shared.ram_cache_reports.append({'partition': partition.name, 'mode': mode,
            'bytes': required, 'whole_partition_CPU_copy': False, 'host_staging_max_bytes': 64 << 20,
            'GPU_joint_budget_bytes': 32 * 2**30, 'forward_reserve_bytes': 14 * 2**30,
            'overlay_sha256': overlay_sha})
        shared.guard.check()

    def batch(partition, indices, keep=None):
        import torch
        if getattr(partition, '_gpu_joints', None) is None:
            result = original_batch(partition, indices, keep)
            release_mapping(partition.joint_present)
            release_mapping(partition.joint_absent)
            return result
        indices = np.asarray(indices, dtype=np.int64)
        if keep is None:
            keep = np.ones(len(indices), dtype=bool)
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != indices.shape:
            raise ValueError('One authoritative evidence mask per row is required')
        shared = partition.shared
        rows = partition.records[indices]
        unique_c, c_inverse = np.unique(rows['target'], return_inverse=True)
        condition = shared.gather('condition', unique_c)
        query = shared.gather('query', np.asarray(rows['query'], dtype=np.int64))
        prior = shared.tensor(partition.cards[partition.card_ids[indices]], floating=True)
        prior_num = np.array(partition.features[indices], dtype=np.float32, copy=True)
        prior_num[~keep] = 0
        prior_num[~keep, :3] = 1 / 3
        prior_num[~keep, -1] = 0
        device = shared.device
        joint_ids = torch.as_tensor(np.array(partition.joint_ids[indices], dtype=np.int64, copy=True), device=device)
        p, a = partition._gpu_joints
        mask = torch.as_tensor(keep, device=device)
        joint = torch.where(mask[:, None, None], p[joint_ids], a[joint_ids]).to(torch.bfloat16)
        if not np.isfinite(prior_num).all() or not torch.isfinite(joint).all():
            raise FloatingPointError('Non-finite cached evidence')
        result = (condition, torch.ones(condition.shape[:2], dtype=torch.bool, device=device),
            shared.tensor(shared.condition_num[unique_c]).float(),
            query, torch.ones(query.shape[:2], dtype=torch.bool, device=device),
            shared.tensor(shared.query_num[rows['query']]).float(), prior,
            torch.as_tensor(keep[:, None].repeat(4, axis=1), device=device), shared.tensor(prior_num).float(),
            torch.as_tensor(c_inverse, device=device, dtype=torch.long), joint,
            torch.ones(joint.shape[:2], dtype=torch.bool, device=device), None)
        # Exact reference replay for a small all-present and a mixed/absent case.
        key = bool(keep.all())
        if key not in partition._parity_checked:
            tiny = min(4, len(indices))
            for test_keep in (keep[:tiny], np.zeros(tiny, dtype=bool), np.ones(tiny, dtype=bool)):
                expected = original_batch(partition, indices[:tiny], test_keep)
                actual = batch_without_parity(partition, indices[:tiny], test_keep)
                if any((x is None) != (y is None) or
                       (x is not None and not torch.equal(x, y)) for x, y in zip(actual, expected)):
                    raise ValueError('GPU cache placement changed A4 input tensors')
            release_mapping(partition.joint_present)
            release_mapping(partition.joint_absent)
            partition._parity_checked.update((True, False))
            for report in shared.ram_cache_reports:
                if report['partition'] == partition.name:
                    report['original_input_tensor_parity'] = 'PASS: all present, all absent, current keep-mask'
        return result

    def batch_without_parity(partition, indices, keep):
        checked = partition._parity_checked
        partition._parity_checked = {True, False}
        try:
            return batch(partition, indices, keep)
        finally:
            partition._parity_checked = checked

    def bindings(*args, **kwargs):
        result = original_bindings(*args, **kwargs)
        result['execution_only_overlay_sha256'] = overlay_sha
        result['scientific_config_unchanged'] = True
        return result

    train.SharedCache.preload_partition = preload
    train.Partition.batch = batch
    train.bindings = bindings
    train.RESTORE_ENTRYPOINT = Path(__file__).resolve()
    _INSTALLED = True


def main(argv=None):
    install()
    return train.main(argv)


if __name__ == '__main__':
    main()
