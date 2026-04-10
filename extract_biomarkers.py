#!/usr/bin/env python3
"""
EEG Biomarker Extraction — shared library

Dataset-agnostic biomarker functions that operate on mne.io.Raw objects.
Import these from dataset-specific scripts (extract_ds004504.py, extract_lead.py).

Biomarkers
----------
Spectral:    band powers, slowing ratio, PDR, IAF, frontal/posterior asymmetry
Connectivity: posterior coherence, global coherence, PLI
Complexity:  permutation entropy, Lempel-Ziv complexity
Derived:     theta/alpha ratio
"""

import warnings
import numpy as np
from scipy.signal import welch, coherence, butter, sosfilt, hilbert

import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')


# ── Constants ──────────────────────────────────────────────────────────────────
BANDS = {
    'delta': (0.5,  4),
    'theta': (4,    8),
    'alpha': (8,   13),
    'beta':  (13,  30),
    'gamma': (30,  45),
}
POSTERIOR_CHANNELS = ['O1', 'O2', 'P3', 'P4', 'Pz']
FRONTAL_CHANNELS   = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']

BIOMARKER_KEYS = [
    'pdr_hz', 'iaf_hz',
    'delta_power', 'theta_power', 'alpha_power', 'beta_power',
    'slowing_ratio', 'theta_alpha_ratio',
    'posterior_coherence', 'global_coherence', 'pli_alpha',
    'frontal_posterior_asym', 'perm_entropy', 'lz_complexity',
]


# ── Spectral biomarkers ────────────────────────────────────────────────────────
def compute_band_powers(raw, window_sec=4, fmin=0.5, fmax=45):
    """
    Relative band power per frequency band, averaged across all channels.

    Note: This computes relative power per channel first, then averages. The alternative — summing across channels before dividing — would compute global relative power where high-amplitude channels dominate the normalization.

    For example, if one channel has 10x higher amplitude than others, it would drag the denominator up and make every band look weaker. The current approach treats each channel equally regardless of its absolute amplitude.

    Parameters
    ----------
    window_sec : float
        Welch window duration in seconds. Determines frequency resolution
        (Δf = 1 / window_sec). Default 4s gives Δf = 0.25 Hz.
    fmin : float
        Lower frequency bound (Hz) for broadband power normalization. Default 0.5.
    fmax : float
        Upper frequency bound (Hz) for broadband power normalization and clipping. Default 45.

    Returns
    -------
    band_powers : dict  {band_name: float}  relative power in [0, 1]
    freqs       : ndarray
    psd_array   : ndarray  (n_channels, n_freqs)
    """
    data  = raw.get_data()
    sfreq = raw.info['sfreq']
    nperseg = int(sfreq * window_sec)
    freqs, _ = welch(data[0], fs=sfreq, nperseg=nperseg)
    psds = []
    # Compute PSD for each channel and store in psd_array (n_channels, n_freqs)
    for ch_data in data:
        _, p = welch(ch_data, fs=sfreq, nperseg=nperseg)
        print(type(p), p.shape)
        psds.append(p)

    # psd array shape: (n_channels, n_freqs)
    psd_array   = np.array(psds)
    broad_mask  = (freqs >= fmin) & (freqs <= fmax)
    # total_power (n_channels,) Total power of all bands for each channel.
    total_power = psd_array[:, broad_mask].sum(axis=1)

    band_powers = {}
    for band, (flo, fhi) in BANDS.items():
        # the mask selects frequencies within the current band.
        mask = (freqs >= flo) & (freqs <= fhi)
        # band_power (n_channels,) total power of `band` for each channel. We sum across frequencies within the band.
        band_power  = psd_array[:, mask].sum(axis=1)
        # band_powers (5,) Division gets relative power per channel. mean() is across channels to get a single value for the band power. `band_powers` is a power for each of the 5 bands
        band_powers[band] = float((band_power / (total_power + 1e-12)).mean())

    return band_powers, freqs, psd_array


def compute_slowing_ratio(band_powers):
    """(delta + theta) / (alpha + beta).  Higher = more slowing."""
    slow = band_powers['delta'] + band_powers['theta']
    fast = band_powers['alpha'] + band_powers['beta']
    return float(slow / (fast + 1e-12))


def compute_theta_alpha_ratio(band_powers):
    """theta / alpha.  Higher = more slowing / cognitive load."""
    return float(band_powers['theta'] / (band_powers['alpha'] + 1e-12))


def compute_pdr(raw, posterior_channels=POSTERIOR_CHANNELS):
    """
    Posterior Dominant Rhythm: spectral peak in 4-14 Hz over posterior
    electrodes.  Clinical norm ~9-10 Hz; values < 8 Hz are 'slowed'.
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names] or raw.ch_names[-4:]
    avg       = raw.copy().pick(available).get_data().mean(axis=0)
    freqs, psd = welch(avg, fs=sfreq, nperseg=int(sfreq * 4))
    mask = (freqs >= 4) & (freqs <= 14)
    return float(freqs[mask][np.argmax(psd[mask])])


def compute_iaf(raw, posterior_channels=POSTERIOR_CHANNELS):
    """
    Individual Alpha Frequency: spectral centre of gravity in 7-13 Hz over
    posterior electrodes.  More stable than PDR (argmax).
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names] or raw.ch_names[-4:]
    avg       = raw.copy().pick(available).get_data().mean(axis=0)
    freqs, psd = welch(avg, fs=sfreq, nperseg=int(sfreq * 4))
    mask   = (freqs >= 7) & (freqs <= 13)
    f_a, p_a = freqs[mask], psd[mask]
    return float(np.sum(f_a * p_a) / (np.sum(p_a) + 1e-12))


def compute_frontal_posterior_asymmetry(raw, frontal_channels=FRONTAL_CHANNELS,
                                        posterior_channels=POSTERIOR_CHANNELS):
    """
    (theta+alpha) power ratio: frontal / posterior.

    >1 = FTD pattern (frontal-dominant slowing).
    <1 = AD pattern  (posterior-dominant slowing).
    """
    sfreq = raw.info['sfreq']

    def _region_power(channels):
        avail = [ch for ch in channels if ch in raw.ch_names]
        if not avail:
            return np.nan
        data = raw.copy().pick(avail).get_data()
        powers = []
        for ch_data in data:
            f, p = welch(ch_data, fs=sfreq, nperseg=int(sfreq * 4))
            powers.append(p[(f >= 4) & (f <= 13)].sum())
        return float(np.mean(powers))

    frontal_p  = _region_power(frontal_channels)
    posterior_p = _region_power(posterior_channels)
    if np.isnan(frontal_p) or np.isnan(posterior_p):
        return np.nan
    return float(frontal_p / (posterior_p + 1e-12))


# ── Connectivity biomarkers ────────────────────────────────────────────────────
def compute_coherence_posterior(raw, posterior_channels=POSTERIOR_CHANNELS):
    """
    Mean magnitude-squared coherence in 8-13 Hz across posterior electrode pairs.
    Reduced in AD (posterior network breakdown).
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names]
    if len(available) < 2:
        return np.nan
    data = raw.copy().pick(available).get_data()
    cohs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            f, coh = coherence(data[i], data[j], fs=sfreq, nperseg=int(sfreq * 4))
            cohs.append(coh[(f >= 8) & (f <= 13)].mean())
    return float(np.mean(cohs))


def compute_coherence_global(raw):
    """
    Mean magnitude-squared coherence in 8-13 Hz across ALL channel pairs.
    Captures whole-brain synchrony.
    """
    sfreq = raw.info['sfreq']
    data  = raw.get_data()
    n_ch  = data.shape[0]
    cohs  = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, coh = coherence(data[i], data[j], fs=sfreq, nperseg=int(sfreq * 4))
            cohs.append(coh[(f >= 8) & (f <= 13)].mean())
    return float(np.mean(cohs))


def compute_pli(raw, band=(8, 13)):
    """
    Mean Phase Lag Index in the alpha band across all channel pairs.
    Robust to volume conduction (unlike magnitude-squared coherence).
    """
    sfreq    = raw.info['sfreq']
    data     = raw.get_data()
    sos      = butter(4, [band[0], band[1]], btype='band', fs=sfreq, output='sos')
    filtered = np.array([sosfilt(sos, ch) for ch in data])
    phases   = np.angle(hilbert(filtered, axis=1))
    n_ch     = data.shape[0]
    pli_vals = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phases[i] - phases[j]
            pli_vals.append(float(np.abs(np.mean(np.sign(np.sin(dphi))))))
    return float(np.mean(pli_vals))


# ── Complexity biomarkers ──────────────────────────────────────────────────────
def compute_permutation_entropy(raw, order=3, delay=1, windowed=False, win_sec=4.0):
    """
    Normalised permutation entropy, averaged across channels.
    Lower = more predictable/regular (AD pattern).

    Parameters
    ----------
    raw      : mne.io.Raw
    order    : Permutation order (default 3).
    delay    : Embedding delay (default 1).
    windowed : If True, compute per non-overlapping window then average.
    win_sec  : Window length in seconds (only used when windowed=True).

    Requires the `antropy` package; returns NaN if unavailable.
    """
    try:
        import antropy as ant
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        if not windowed:
            return float(np.mean([
                ant.perm_entropy(ch, order=order, delay=delay, normalize=True)
                for ch in data
            ]))

        win_samples = int(sfreq * win_sec)
        n_samples = data.shape[1]
        if n_samples < win_samples:
            return float(np.mean([
                ant.perm_entropy(ch, order=order, delay=delay, normalize=True)
                for ch in data
            ]))

        pe_per_ch = []
        for ch in data:
            wins = [ch[i:i + win_samples] for i in range(0, n_samples - win_samples + 1, win_samples)]
            pe_per_ch.append(np.mean([
                ant.perm_entropy(w, order=order, delay=delay, normalize=True)
                for w in wins
            ]))
        return float(np.mean(pe_per_ch))

    except ImportError:
        return np.nan


def compute_lempel_ziv(raw, windowed=False, win_sec=4.0):
    """
    Normalised Lempel-Ziv complexity on the binarised signal (median threshold).
    Lower = more regular/repetitive (AD pattern).

    Parameters
    ----------
    raw      : mne.io.Raw
    windowed : If True, compute per non-overlapping window then average.
               More robust to signal length and respects quasi-stationarity.
    win_sec  : Window length in seconds (only used when windowed=True).

    Requires the `antropy` package; returns NaN if unavailable.
    """
    try:
        import antropy as ant
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        if not windowed:
            return float(np.mean([
                ant.lziv_complexity((ch > np.median(ch)).astype(int), normalize=True)
                for ch in data
            ]))

        win_samples = int(sfreq * win_sec)
        n_samples = data.shape[1]
        if n_samples < win_samples:
            # Signal shorter than one window — fall back to whole-signal
            return float(np.mean([
                ant.lziv_complexity((ch > np.median(ch)).astype(int), normalize=True)
                for ch in data
            ]))

        lzc_per_ch = []
        for ch in data:
            wins = [ch[i:i + win_samples] for i in range(0, n_samples - win_samples + 1, win_samples)]
            lzc_per_ch.append(np.mean([
                ant.lziv_complexity((w > np.median(w)).astype(int), normalize=True)
                for w in wins
            ]))
        return float(np.mean(lzc_per_ch))

    except ImportError:
        return np.nan


# ── Convenience: extract all biomarkers from a single Raw ──────────────────────
def extract_subject_biomarkers(raw, windowed_complexity=False) -> dict:
    """
    Run all 11 biomarker extractions on one mne.io.Raw.

    Parameters
    ----------
    raw                 : mne.io.Raw
    windowed_complexity : If True, compute LZC and permutation entropy per
                          window then average.  Better for long or concatenated
                          signals; unnecessary if the input is already a short
                          pre-segmented window.

    Returns a dict with keys matching BIOMARKER_KEYS plus individual band powers.
    Caller is responsible for adding metadata (subject, group, age, etc.).
    """
    bp, _, _ = compute_band_powers(raw)
    rec = {
        'theta_alpha_ratio':      compute_theta_alpha_ratio(bp),
        'pdr_hz':                 compute_pdr(raw),
        'iaf_hz':                 compute_iaf(raw),
        'slowing_ratio':          compute_slowing_ratio(bp),
        'posterior_coherence':     compute_coherence_posterior(raw),
        'global_coherence':       compute_coherence_global(raw),
        'pli_alpha':              compute_pli(raw),
        'frontal_posterior_asym':  compute_frontal_posterior_asymmetry(raw),
        'perm_entropy':           compute_permutation_entropy(raw, windowed=windowed_complexity),
        'lz_complexity':          compute_lempel_ziv(raw, windowed=windowed_complexity),
        **{f'{band}_power': val for band, val in bp.items()},
    }
    return rec
