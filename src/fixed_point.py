"""Bit-accurate, signed two's-complement fixed-point linear inference.

Inputs are SELECTED, IMPUTED, STANDARDIZED sensor values, not raw sensors.
The RTL uses the same integer dot product and offset comparison as here.
"""
import math
import numpy as np


def fractional_bits(max_abs: float, bits: int) -> int:
    if bits < 4 or bits > 32:
        raise ValueError('bits must be in [4, 32]')
    if not np.isfinite(max_abs) or max_abs < 0:
        raise ValueError('invalid magnitude')
    limit = 2 ** (bits - 1) - 1
    if max_abs == 0:
        return bits - 2
    return max(0, min(bits - 2, int(math.floor(math.log2(limit / max_abs)))))


def quantize(values, bits: int, frac: int) -> np.ndarray:
    if frac < 0 or frac >= bits:
        raise ValueError('invalid fractional bits')
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('values must be finite')
    return np.clip(np.rint(x * (2 ** frac)), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1).astype(np.int64)


def quantized_linear_predict(x_int, w_int, offset_int):
    """Exact 64-bit integer arithmetic for bounded 8/12/16-bit vectors."""
    x = np.asarray(x_int, dtype=np.int64)
    w = np.asarray(w_int, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 1 or x.shape[1] != len(w):
        raise ValueError('bad matrix/weight dimensions')
    return (x @ w + np.int64(offset_int)) >= 0


def sequential_mac_predict(x_int, w_int, offset_int):
    """Bit-accurate reference for the sequential MAC RTL.

    Accumulates one feature*weight product at a time starting from offset_int,
    matching the exact arithmetic of hardware/linear_classifier_seq.sv.
    The RTL posts this decision N cycles after accepting a sample and can
    accept the next sample on that same cycle.
    """
    x = np.asarray(x_int, dtype=np.int64)
    w = np.asarray(w_int, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 1 or x.shape[1] != len(w):
        raise ValueError('bad matrix/weight dimensions')
    # Start the 64-bit accumulator at the quantized bias/threshold offset.
    acc = np.full(x.shape[0], np.int64(offset_int), dtype=np.int64)
    for index in range(len(w)):
        acc = acc + x[:, index] * np.int64(w[index])
    return acc >= 0


def make_fixed_model(model, train_matrix, bits: int, threshold: float):
    if model.coef_.shape[0] != 1:
        raise ValueError('binary linear classifier expected')
    if not 0 < threshold < 1:
        raise ValueError('probability threshold must be in (0, 1)')
    weights = model.coef_[0]
    xfrac = fractional_bits(float(np.max(np.abs(train_matrix))), bits)
    wfrac = fractional_bits(float(np.max(np.abs(weights))), bits)
    # compare dot(x, w) + (intercept - logit(threshold)) >= 0
    offset = int(np.rint((float(model.intercept_[0]) - math.log(threshold / (1 - threshold))) * (2 ** (xfrac + wfrac))))
    w_int = quantize(weights, bits, wfrac)
    if len(w_int) * (2 ** (bits - 1)) ** 2 + abs(offset) >= 2 ** 63:
        raise OverflowError('int64 accumulator cannot hold worst-case sum')
    return dict(bits=bits, xfrac=xfrac, wfrac=wfrac, weights=w_int.tolist(), offset=offset, threshold=float(threshold))


def to_twos_hex(value: int, bits: int) -> str:
    if not -(2 ** (bits - 1)) <= int(value) < 2 ** (bits - 1):
        raise ValueError('outside signed range')
    return f'{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}'
