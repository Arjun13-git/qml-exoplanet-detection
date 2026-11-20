"""
SNR-controlled Gaussian noise injection for EVALUATION-ONLY stress testing.

This module is intentionally separate from the training-time augmentation in
``src/data/dataset.py`` (fixed sigma = 0.001, applied to the training split only).
That augmentation is not used, imported, or modified here.

Definition (per sample, over all feature dimensions of that sample)
------------------------------------------------------------------
    P_signal = Var(x)            if power_mode == "variance"     (default; mean-removed / AC power)
             = mean(x ** 2)      if power_mode == "mean_square"  (total power, includes DC offset)
    P_noise  = P_signal / SNR    (SNR is linear, NOT decibels)
    sigma    = sqrt(P_noise)
    x_noisy  = x + n,  n ~ N(0, sigma^2), i.i.d. (white, uncorrelated) across features and samples

The functions never modify their input arrays; they always return new arrays.
Passing ``snr=None`` returns an unmodified copy (clean / noise-disabled path).
"""
import numpy as np

DEFAULT_SNR_LEVELS = (10.0, 5.0, 1.0, 0.1)
POWER_MODES = ("variance", "mean_square")


def _as_2d(x):
    """View any (n_samples, ...) array as (n_samples, n_features) float64."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("Input must have at least one dimension.")
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


def signal_power(x, power_mode="variance"):
    """Per-sample signal power, shape (n_samples,)."""
    if power_mode not in POWER_MODES:
        raise ValueError(f"power_mode must be one of {POWER_MODES}, got {power_mode!r}")
    x2 = _as_2d(x)
    if power_mode == "variance":
        return x2.var(axis=1)
    return np.mean(x2 ** 2, axis=1)


def noise_std_for_snr(power, snr):
    """sigma = sqrt(P_signal / SNR) for a linear SNR > 0."""
    snr = float(snr)
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError(f"SNR must be a finite positive number (linear scale), got {snr}")
    return np.sqrt(np.asarray(power, dtype=np.float64) / snr)


def add_gaussian_noise_snr(x, snr, seed=None, rng=None, power_mode="variance", return_std=False):
    """
    Return a NEW array equal to ``x`` plus white Gaussian noise at the requested linear SNR.

    Args:
        x: array of shape (n_samples, ...) - the actual model input representation.
        snr: linear SNR (e.g. 10, 5, 1, 0.1). ``None`` disables noise (returns a copy of x).
        seed: integer seed for reproducibility (ignored if ``rng`` is given).
        rng: optional ``numpy.random.Generator``.
        power_mode: "variance" (default) or "mean_square".
        return_std: if True, also return the per-sample noise std actually used.
    """
    x_in = np.asarray(x)
    out_dtype = x_in.dtype if np.issubdtype(x_in.dtype, np.floating) else np.float64

    if snr is None:
        clean = np.array(x_in, dtype=out_dtype, copy=True)
        return (clean, np.zeros(_as_2d(x_in).shape[0])) if return_std else clean

    x2 = _as_2d(x_in)
    sigma = noise_std_for_snr(signal_power(x2, power_mode), snr)
    if rng is None:
        rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x2.shape) * sigma[:, None]
    noisy = (x2 + noise).reshape(x_in.shape).astype(out_dtype, copy=False)
    return (noisy, sigma) if return_std else noisy


def empirical_snr(clean, noisy, power_mode="variance"):
    """Measured per-sample SNR = P_signal(clean) / mean((noisy - clean)^2)."""
    c2, n2 = _as_2d(clean), _as_2d(noisy)
    noise_power = np.mean((n2 - c2) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return signal_power(c2, power_mode) / noise_power
