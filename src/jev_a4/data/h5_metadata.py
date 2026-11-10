"""Decode both legacy and current AnnData string/categorical columns."""
from __future__ import annotations

import h5py
import numpy as np


def _strings(dataset):
    if h5py.check_string_dtype(dataset.dtype):
        return np.asarray(dataset.asstr()[:], dtype=str)
    return np.asarray(dataset[:]).astype(str)


def text_array(group, name, allow_missing=False):
    if name not in group:
        if allow_missing:
            return None
        raise KeyError(name)
    column = group[name]
    categories = None
    if isinstance(column, h5py.Group):
        if not {"categories", "codes"}.issubset(column.keys()):
            raise ValueError(f"Unsupported encoded column: {column.name}")
        categories = column["categories"]
        codes = np.asarray(column["codes"][:])
    elif "categories" in column.attrs:
        reference = column.attrs["categories"]
        if isinstance(reference, bytes):
            reference = reference.decode()
        if isinstance(reference, h5py.Reference):
            categories = column.file[reference]
        else:
            categories = group[reference]
        codes = np.asarray(column[:])
    elif "__categories" in group and name in group["__categories"]:
        categories = group["__categories"][name]
        codes = np.asarray(column[:])
    else:
        return _strings(column), None
    values = _strings(categories)
    if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"Non-integer categorical codes: {column.name}")
    if np.any(codes < -1) or np.any(codes >= len(values)):
        raise ValueError(f"Categorical code outside its vocabulary: {column.name}")
    decoded = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    decoded[valid] = values[codes[valid]]
    return decoded.astype(str), codes
