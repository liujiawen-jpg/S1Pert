from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from jev_a4.labels.build_resolution_bundle import IGNORE_INDEX, artifact_identity, map_arrays
except ModuleNotFoundError as error:
    # The label bundle copies these small audit scripts into its code/ folder.
    if error.name != "jev_a4":
        raise
    from build_resolution_bundle import IGNORE_INDEX, artifact_identity, map_arrays


def test_resolved_and_unresolved_are_never_conflated():
    labels = np.asarray([0, 1, 2, -1, -1, -1, -1, -1], dtype=np.int8)
    reasons = np.asarray([0, 0, 0, 1, 2, 3, 4, 5], dtype=np.uint8)
    effects = np.asarray([0.5, 0.0, -0.5, np.nan, 0.1, np.nan, 0.2, -0.4])
    hard, mask, resolution, point = map_arrays(
        labels, reasons, effects, log2_margin=float(np.log2(1.2))
    )
    assert hard.tolist() == [0, 1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert mask.tolist() == [True, True, True, False, False, False, False, False]
    assert resolution.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert point.tolist() == [0, 1, 2, -1, 1, -1, 1, 2]


def test_unresolved_reason_zero_is_rejected():
    with pytest.raises(ValueError, match="resolved iff"):
        map_arrays(
            np.asarray([-1], dtype=np.int8),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0.0]),
            log2_margin=float(np.log2(1.2)),
        )


def test_resolved_reason_nonzero_is_rejected():
    with pytest.raises(ValueError, match="resolved iff"):
        map_arrays(
            np.asarray([1], dtype=np.int8),
            np.asarray([5], dtype=np.uint8),
            np.asarray([0.0]),
            log2_margin=float(np.log2(1.2)),
        )


def test_point_region_is_not_allowed_to_contradict_strict_label():
    with pytest.raises(ValueError, match="effect regions"):
        map_arrays(
            np.asarray([0], dtype=np.int8),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0.0]),
            log2_margin=float(np.log2(1.2)),
        )


def test_unexpected_label_and_reason_codes_are_rejected():
    with pytest.raises(ValueError, match="label code"):
        map_arrays(np.asarray([3]), np.asarray([0]), np.asarray([1.0]), log2_margin=0.1)
    with pytest.raises(ValueError, match="reason code"):
        map_arrays(np.asarray([-1]), np.asarray([6]), np.asarray([1.0]), log2_margin=0.1)


def test_artifact_identity_uses_movable_relative_path(tmp_path):
    artifact = tmp_path / "array.npy"
    artifact.write_bytes(b"values")
    identity = artifact_identity(artifact, "array.npy")
    assert identity["path"] == "array.npy"
    assert identity["size_bytes"] == 6
