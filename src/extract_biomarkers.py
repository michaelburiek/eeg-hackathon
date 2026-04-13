#!/usr/bin/env python3
"""
EEG Biomarker Extraction — shared library

Dataset-agnostic biomarker functions that operate on mne.io.Raw objects.
Import these from dataset-specific scripts (scripts/biomarkers/).

Biomarkers
----------
Spectral:    band powers, slowing ratio, TAR, PDR, IAF, frontal/posterior asymmetry,
             aperiodic exponent (1/f slope via specparam)
Connectivity: posterior coherence, global coherence, PLI, PAC
Complexity:  permutation entropy, Lempel-Ziv complexity, sample entropy,
             spectral entropy
"""

import warnings
import numpy as np
from scipy.signal import welch, coherence, butter, sosfilt, sosfiltfilt, hilbert

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
    'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
    'slowing_ratio', 'theta_alpha_ratio',
    'posterior_coherence', 'global_coherence', 'pli_alpha',
    'frontal_posterior_asym', 'perm_entropy', 'lz_complexity',
    'sample_entropy', 'spectral_entropy',
    'aperiodic_exponent',
    'pac_theta_gamma', 'pac_delta_alpha',
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
    freqs       : ndarray (n_freqs,) frequencies corresponding to PSD columns
    psd_array   : ndarray  (n_channels, n_freqs) PSD power values, for each channel
    """
    data  = raw.get_data()
    sfreq = raw.info['sfreq']
    nperseg = int(sfreq * window_sec)
    freqs, _ = welch(data[0], fs=sfreq, nperseg=nperseg)
    psds = []
    # Compute PSD for each channel and store in psd_array (n_channels, n_freqs)
    for ch_data in data:
        _, p = welch(ch_data, fs=sfreq, nperseg=nperseg)
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
    """
    (delta + theta) / (alpha + beta).

    Returns
    -------
    float : ratio > 1 indicates dominant slow-wave activity, a hallmark of
            cortical slowing seen in AD and other dementias.
    """
    slow = band_powers['delta'] + band_powers['theta']
    fast = band_powers['alpha'] + band_powers['beta']
    return float(slow / (fast + 1e-12))


def compute_theta_alpha_ratio(band_powers):
    """
    theta / alpha.

    Returns
    -------
    float : ratio of theta to alpha power. Elevated values indicate alpha
            rhythm degradation with compensatory theta increase, commonly
            observed in early-stage AD and cognitive decline.
    """
    return float(band_powers['theta'] / (band_powers['alpha'] + 1e-12))


def compute_pdr(raw, posterior_channels=POSTERIOR_CHANNELS, fmin=4, fmax=14, window_sec=4):
    """
    Posterior Dominant Rhythm: spectral peak in fmin-fmax Hz over posterior
    electrodes.

    Returns
    -------
    float : peak frequency (Hz) in the posterior region. Clinical norm is
            ~9-10 Hz; values < 8 Hz indicate slowed background activity,
            a common early sign of AD.
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names] or raw.ch_names[-4:]
    avg       = raw.copy().pick(available).get_data().mean(axis=0)
    freqs, psd = welch(avg, fs=sfreq, nperseg=int(sfreq * window_sec))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(freqs[mask][np.argmax(psd[mask])])


def compute_iaf(raw, posterior_channels=POSTERIOR_CHANNELS, fmin=7, fmax=13, window_sec=4):
    """
    Individual Alpha Frequency: power-weighted mean frequency in fmin-fmax Hz
    over posterior electrodes. More stable than PDR (argmax) because it uses
    centre of gravity instead of a single peak.

    Returns
    -------
    float : centre-of-gravity frequency (Hz) in the alpha range. Healthy
            adults typically show ~10 Hz; lower values correlate with
            cognitive impairment severity.
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names] or raw.ch_names[-4:]
    avg       = raw.copy().pick(available).get_data().mean(axis=0)
    freqs, psd = welch(avg, fs=sfreq, nperseg=int(sfreq * window_sec))
    mask   = (freqs >= fmin) & (freqs <= fmax)
    f_a, p_a = freqs[mask], psd[mask]
    return float(np.sum(f_a * p_a) / (np.sum(p_a) + 1e-12))


def compute_frontal_posterior_asymmetry(raw, frontal_channels=FRONTAL_CHANNELS,
                                        posterior_channels=POSTERIOR_CHANNELS,
                                        window_sec=4, fmin=4, fmax=13):
    """
    (theta+alpha) power ratio: frontal / posterior.

    Returns
    -------
    float : ratio of frontal to posterior slow-band power.
            >1 = FTD pattern (frontal-dominant slowing).
            <1 = AD pattern (posterior-dominant slowing).
            Useful for differentiating AD from FTD.
    """
    sfreq = raw.info['sfreq']

    def _region_power(channels):
        avail = [ch for ch in channels if ch in raw.ch_names]
        if not avail:
            return np.nan
        data = raw.copy().pick(avail).get_data()
        powers = []
        for ch_data in data:
            f, p = welch(ch_data, fs=sfreq, nperseg=int(sfreq * window_sec))
            powers.append(p[(f >= fmin) & (f <= fmax)].sum())
        return float(np.mean(powers))

    frontal_p  = _region_power(frontal_channels)
    posterior_p = _region_power(posterior_channels)
    if np.isnan(frontal_p) or np.isnan(posterior_p):
        return np.nan
    return float(frontal_p / (posterior_p + 1e-12))


def compute_aperiodic_exponent(raw, freq_range=(1, 40)):
    """
    Aperiodic (1/f) exponent from the specparam (FOOOF) model, fit to the
    mean PSD across all channels.

    Returns
    -------
    float : aperiodic exponent (slope of the 1/f background in log-log space).
            Higher values (~1.5+) indicate steeper spectral falloff, seen in AD.
            Lower values (~1.0) are typical of healthy adults.
            Requires the `specparam` package; returns NaN if unavailable.
    """
    try:
        from specparam import SpectralModel
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        nperseg = min(int(sfreq * 4), data.shape[1])
        freqs, _ = welch(data[0], fs=sfreq, nperseg=nperseg)
        psds = []
        for ch in data:
            _, p = welch(ch, fs=sfreq, nperseg=nperseg)
            psds.append(p)
        mean_psd = np.mean(psds, axis=0)

        fm = SpectralModel(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=6,
            min_peak_height=0.1,
            verbose=False,
        )
        fm.fit(freqs, mean_psd, freq_range=freq_range)
        ap = fm.get_params('aperiodic')
        return float(ap[1])  # exponent
    except ImportError:
        return np.nan


# ── Connectivity biomarkers ────────────────────────────────────────────────────
def compute_coherence_posterior(raw, posterior_channels=POSTERIOR_CHANNELS,
                                window_sec=4, fmin=8, fmax=13):
    """
    Mean magnitude-squared coherence in fmin-fmax Hz across posterior electrode pairs.

    Returns
    -------
    float : mean coherence [0, 1] across posterior channel pairs. Reduced
            values indicate posterior network breakdown, characteristic of AD.
    """
    sfreq     = raw.info['sfreq']
    available = [ch for ch in posterior_channels if ch in raw.ch_names]
    if len(available) < 2:
        return np.nan
    data = raw.copy().pick(available).get_data()
    cohs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            f, coh = coherence(data[i], data[j], fs=sfreq, nperseg=int(sfreq * window_sec))
            cohs.append(coh[(f >= fmin) & (f <= fmax)].mean())
    return float(np.mean(cohs))


def compute_coherence_global(raw, window_sec=4, fmin=8, fmax=13):
    """
    Mean magnitude-squared coherence in fmin-fmax Hz across ALL channel pairs.

    Returns
    -------
    float : mean coherence [0, 1] across all channel pairs. Captures
            whole-brain synchrony; reduced in widespread neurodegeneration.
    """
    sfreq = raw.info['sfreq']
    data  = raw.get_data()
    n_ch  = data.shape[0]
    cohs  = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, coh = coherence(data[i], data[j], fs=sfreq, nperseg=int(sfreq * window_sec))
            cohs.append(coh[(f >= fmin) & (f <= fmax)].mean())
    return float(np.mean(cohs))


def compute_pli(raw, band=(8, 13), filter_order=4):
    """
    Mean Phase Lag Index in the alpha band across all channel pairs.
    Robust to volume conduction (unlike magnitude-squared coherence).

    Returns
    -------
    float : mean PLI [0, 1] across all channel pairs. Measures phase
            synchronization while rejecting zero-lag (volume-conducted)
            signals. Reduced in AD due to cortical disconnection.
    """
    sfreq    = raw.info['sfreq']
    data     = raw.get_data()
    sos      = butter(filter_order, [band[0], band[1]], btype='band', fs=sfreq, output='sos')
    filtered = np.array([sosfilt(sos, ch) for ch in data])
    phases   = np.angle(hilbert(filtered, axis=1))
    n_ch     = data.shape[0]
    pli_vals = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phases[i] - phases[j]
            pli_vals.append(float(np.abs(np.mean(np.sign(np.sin(dphi))))))
    return float(np.mean(pli_vals))


TEMPORAL_CHANNELS = ['T3', 'T4', 'T5', 'T6', 'C3', 'C4', 'Cz']


def compute_pac(raw, low_band=(4, 8), high_band=(30, 45), n_bins=18,
                temporal_channels=TEMPORAL_CHANNELS):
    """
    Phase-Amplitude Coupling via Modulation Index (Tort et al., 2010) on a
    temporal/central channel.

    Returns
    -------
    float : Modulation Index [0, 1]. Measures how strongly the amplitude of
            high-frequency oscillations is modulated by the phase of
            low-frequency oscillations. Reduced theta-gamma PAC is
            associated with memory impairment in AD.
    """
    sfreq = raw.info['sfreq']
    ch_use = next((c for c in temporal_channels if c in raw.ch_names),
                  raw.ch_names[min(8, len(raw.ch_names) - 1)])
    ch_idx = raw.ch_names.index(ch_use)
    data = raw.get_data()[ch_idx]

    sos_low = butter(4, low_band, btype='band', fs=sfreq, output='sos')
    sig_low = sosfiltfilt(sos_low, data)

    sos_high = butter(4, high_band, btype='band', fs=sfreq, output='sos')
    sig_high = sosfiltfilt(sos_high, data)

    phase = np.angle(hilbert(sig_low))
    amplitude = np.abs(hilbert(sig_high))

    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    amp_by_phase = np.zeros(n_bins)
    for k in range(n_bins):
        in_bin = (phase >= phase_bins[k]) & (phase < phase_bins[k + 1])
        amp_by_phase[k] = amplitude[in_bin].mean() if in_bin.sum() > 0 else 0

    p = amp_by_phase / (amp_by_phase.sum() + 1e-15)
    q = np.ones(n_bins) / n_bins
    kl_div = np.sum(p * np.log((p + 1e-15) / (q + 1e-15)))
    mi = kl_div / np.log(n_bins)

    return float(mi)


# ── Complexity biomarkers ──────────────────────────────────────────────────────
def compute_permutation_entropy(raw, order=3, delay=1, windowed=False, win_sec=4.0):
    """
    Normalised permutation entropy, averaged across channels.

    Returns
    -------
    float : mean normalised permutation entropy [0, 1]. Lower values
            indicate more predictable/regular signal dynamics, a pattern
            seen in AD reflecting reduced cortical complexity.

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

    Returns
    -------
    float : mean normalised LZ complexity [0, 1]. Lower values indicate
            more regular/repetitive signal patterns, associated with AD
            and reduced neural information processing capacity.

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


def compute_sample_entropy(raw, order=2, r_factor=0.15):
    """
    Sample entropy, averaged across channels. Uses r = r_factor * std(channel)
    as the similarity tolerance.

    Returns
    -------
    float : mean sample entropy across channels. Lower values indicate more
            self-similar, predictable dynamics — characteristic of AD.
            Requires the `antropy` package; returns NaN if unavailable.
    """
    try:
        import antropy as ant
        data = raw.get_data()
        vals = [ant.sample_entropy(ch, order=order) for ch in data]
        return float(np.nanmean(vals))
    except ImportError:
        return np.nan


def compute_spectral_entropy(raw):
    """
    Normalised spectral entropy (Welch method), averaged across channels.

    Returns
    -------
    float : mean normalised spectral entropy [0, 1]. Measures the flatness
            of the power spectrum. Lower values indicate power concentrated
            in fewer bands (more regular oscillations), seen in AD.
            Requires the `antropy` package; returns NaN if unavailable.
    """
    try:
        import antropy as ant
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        vals = [ant.spectral_entropy(ch, sf=sfreq, method='welch', normalize=True)
                for ch in data]
        return float(np.mean(vals))
    except ImportError:
        return np.nan


# ── Convenience: extract all biomarkers from a single Raw ──────────────────────
def extract_subject_biomarkers(raw, windowed_complexity=False) -> dict:
    """
    Run all biomarker extractions on one mne.io.Raw.

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
        'sample_entropy':         compute_sample_entropy(raw),
        'spectral_entropy':       compute_spectral_entropy(raw),
        'aperiodic_exponent':     compute_aperiodic_exponent(raw),
        'pac_theta_gamma':        compute_pac(raw, low_band=(4, 8), high_band=(30, 45)),
        'pac_delta_alpha':        compute_pac(raw, low_band=(1, 4), high_band=(8, 13)),
        **{f'{band}_power': val for band, val in bp.items()},
    }
    return rec
