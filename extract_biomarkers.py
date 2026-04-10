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
import pandas as pd
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
    rec['theta_alpha_ratio'] = bp['theta'] / (bp['alpha'] + 1e-12)
    return rec


# ── CN norms & z-scores ───────────────────────────────────────────────────────
def compute_cn_norms(biomarkers_df: pd.DataFrame) -> dict:
    """Compute mean/std for each biomarker using the CN (healthy control) group."""
    cn = biomarkers_df[biomarkers_df['group'] == 'CN']
    norms = {}
    for k in BIOMARKER_KEYS:
        norms[f'{k}_mean'] = float(cn[k].mean())
        norms[f'{k}_std']  = float(cn[k].std())
    return norms


def append_zscores(df: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """Add z-score columns to a biomarkers DataFrame using precomputed CN norms."""
    df = df.copy()
    for key in BIOMARKER_KEYS:
        df[f'{key}_z'] = (df[key] - norms[f'{key}_mean']) / (norms[f'{key}_std'] + 1e-9)
    return df


def generate_biomarker_summary(row, norms):
    """
    Compute z-scores and generate a human-readable text summary for one subject.

    Parameters
    ----------
    row   : dict or pd.Series with all biomarker fields
    norms : dict with {biomarker}_mean / {biomarker}_std entries (from CN group)

    Returns
    -------
    zscores : dict
    quant   : dict  (actual values with units)
    text    : str
    """
    def _z(val, key):
        return (val - norms[f'{key}_mean']) / (norms[f'{key}_std'] + 1e-9)

    def _arrow(z):
        if z >  1.0: return '\u2191 elevated'
        if z < -1.0: return '\u2193 reduced'
        return '~ normal'

    pdr_z   = _z(row['pdr_hz'],                'pdr_hz')
    iaf_z   = _z(row['iaf_hz'],                'iaf_hz')
    delta_z = _z(row['delta_power'],            'delta_power')
    theta_z = _z(row['theta_power'],            'theta_power')
    alpha_z = _z(row['alpha_power'],            'alpha_power')
    beta_z  = _z(row['beta_power'],             'beta_power')
    sr_z    = _z(row['slowing_ratio'],          'slowing_ratio')
    ta_z    = _z(row['theta_alpha_ratio'],      'theta_alpha_ratio')
    coh_z   = _z(row['posterior_coherence'],    'posterior_coherence')
    gcoh_z  = _z(row['global_coherence'],       'global_coherence')
    pli_z   = _z(row['pli_alpha'],              'pli_alpha')
    fpa_z   = _z(row['frontal_posterior_asym'],  'frontal_posterior_asym')
    pe_z    = _z(row['perm_entropy'],           'perm_entropy')
    lzc_z   = _z(row['lz_complexity'],          'lz_complexity')

    zscores = {
        'pdr_z':                   round(float(pdr_z),   2),
        'iaf_z':                   round(float(iaf_z),   2),
        'delta_power_z':           round(float(delta_z), 2),
        'theta_power_z':           round(float(theta_z), 2),
        'alpha_power_z':           round(float(alpha_z), 2),
        'beta_power_z':            round(float(beta_z),  2),
        'slowing_ratio_z':         round(float(sr_z),    2),
        'theta_alpha_ratio_z':     round(float(ta_z),    2),
        'posterior_coherence_z':    round(float(coh_z),   2),
        'global_coherence_z':      round(float(gcoh_z),  2),
        'pli_alpha_z':             round(float(pli_z),   2),
        'frontal_posterior_asym_z': round(float(fpa_z),   2),
        'perm_entropy_z':          round(float(pe_z),    2),
        'lz_complexity_z':         round(float(lzc_z),   2),
        'PDR_clinically_slowed':   bool(row['pdr_hz'] < 8.0),
    }

    quant = {
        'pdr_hz':                round(float(row['pdr_hz']),                2),
        'iaf_hz':                round(float(row['iaf_hz']),                2),
        'delta_power_pct':       round(float(row['delta_power']) * 100,     1),
        'theta_power_pct':       round(float(row['theta_power']) * 100,     1),
        'alpha_power_pct':       round(float(row['alpha_power']) * 100,     1),
        'beta_power_pct':        round(float(row['beta_power'])  * 100,     1),
        'slowing_ratio':         round(float(row['slowing_ratio']),         3),
        'theta_alpha_ratio':     round(float(row['theta_alpha_ratio']),     3),
        'posterior_coherence':    round(float(row['posterior_coherence']),   3),
        'global_coherence':      round(float(row['global_coherence']),      3),
        'pli_alpha':             round(float(row['pli_alpha']),             3),
        'frontal_posterior_asym': round(float(row['frontal_posterior_asym']), 3),
        'perm_entropy':          round(float(row['perm_entropy']),          4),
        'lz_complexity':         round(float(row['lz_complexity']),         4),
    }

    text = (
        f"Subject: {row['subject']}  |  Age: {row.get('age', 'N/A')}  |  MMSE: {row.get('mmse', 'N/A')}\n"
        f"Group: {row['group']}\n"
        f"\n--- QUANTITATIVE BIOMARKERS ---\n"
        f"Posterior Dominant Rhythm:    {quant['pdr_hz']:.2f} Hz"
        f"  (CN norm: {norms['pdr_hz_mean']:.1f} \u00b1 {norms['pdr_hz_std']:.1f} Hz)"
        f"  [clinically slowed: {'YES' if row['pdr_hz'] < 8.0 else 'no'}]\n"
        f"Individual Alpha Frequency:  {quant['iaf_hz']:.2f} Hz"
        f"  (CN norm: {norms['iaf_hz_mean']:.1f} \u00b1 {norms['iaf_hz_std']:.1f} Hz)\n"
        f"Delta power:                 {quant['delta_power_pct']:.1f}%\n"
        f"Theta power:                 {quant['theta_power_pct']:.1f}%\n"
        f"Alpha power:                 {quant['alpha_power_pct']:.1f}%\n"
        f"Beta power:                  {quant['beta_power_pct']:.1f}%\n"
        f"Slowing ratio (d+t)/(a+b):   {quant['slowing_ratio']:.3f}"
        f"  (CN norm: {norms['slowing_ratio_mean']:.3f})\n"
        f"Theta/Alpha ratio:           {quant['theta_alpha_ratio']:.3f}"
        f"  (CN norm: {norms['theta_alpha_ratio_mean']:.3f})\n"
        f"Posterior alpha coherence:    {quant['posterior_coherence']:.3f}"
        f"  (CN norm: {norms['posterior_coherence_mean']:.3f})\n"
        f"Global alpha coherence:      {quant['global_coherence']:.3f}"
        f"  (CN norm: {norms['global_coherence_mean']:.3f})\n"
        f"Phase Lag Index (alpha):      {quant['pli_alpha']:.3f}"
        f"  (CN norm: {norms['pli_alpha_mean']:.3f})\n"
        f"Frontal/Posterior asymmetry:  {quant['frontal_posterior_asym']:.3f}"
        f"  (CN norm: {norms['frontal_posterior_asym_mean']:.3f})\n"
        f"Permutation entropy:         {quant['perm_entropy']:.4f}"
        f"  (CN norm: {norms['perm_entropy_mean']:.4f})\n"
        f"Lempel-Ziv complexity:       {quant['lz_complexity']:.4f}"
        f"  (CN norm: {norms['lz_complexity_mean']:.4f})\n"
        f"\n--- Z-SCORES vs CN NORMS ---\n"
        f"{'PDR':38s}  z = {pdr_z:+.2f}  {_arrow(pdr_z)}\n"
        f"{'Individual Alpha Frequency':38s}  z = {iaf_z:+.2f}  {_arrow(iaf_z)}\n"
        f"{'Delta power':38s}  z = {delta_z:+.2f}  {_arrow(delta_z)}\n"
        f"{'Theta power':38s}  z = {theta_z:+.2f}  {_arrow(theta_z)}\n"
        f"{'Alpha power':38s}  z = {alpha_z:+.2f}  {_arrow(alpha_z)}\n"
        f"{'Beta power':38s}  z = {beta_z:+.2f}  {_arrow(beta_z)}\n"
        f"{'Slowing ratio':38s}  z = {sr_z:+.2f}  {_arrow(sr_z)}\n"
        f"{'Theta/Alpha ratio':38s}  z = {ta_z:+.2f}  {_arrow(ta_z)}\n"
        f"{'Posterior coherence':38s}  z = {coh_z:+.2f}  {_arrow(coh_z)}\n"
        f"{'Global coherence':38s}  z = {gcoh_z:+.2f}  {_arrow(gcoh_z)}\n"
        f"{'Phase Lag Index (alpha)':38s}  z = {pli_z:+.2f}  {_arrow(pli_z)}\n"
        f"{'Frontal/Posterior asymmetry':38s}  z = {fpa_z:+.2f}  {_arrow(fpa_z)}\n"
        f"{'Permutation entropy':38s}  z = {pe_z:+.2f}  {_arrow(pe_z)}\n"
        f"{'Lempel-Ziv complexity':38s}  z = {lzc_z:+.2f}  {_arrow(lzc_z)}\n"
    )

    return zscores, quant, text
