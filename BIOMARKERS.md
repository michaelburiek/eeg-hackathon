# EEG Biomarker Reference

| Category | Biomarker | Function | What It Captures | AD Pattern | FTD Pattern |
|---|---|---|---|---|---|
| **Spectral: Frequency** | Posterior Dominant Rhythm (PDR) | `compute_pdr` | Peak frequency in 4–14 Hz over posterior electrodes (argmax) | Slowed (<8 Hz) | Mildly slowed |
| **Spectral: Frequency** | Individual Alpha Frequency (IAF) | `compute_iaf` | Center of gravity in 7–13 Hz over posterior electrodes — more stable than PDR | Shifted downward | Shifted downward |
| **Spectral: Power** | Band powers (delta, theta, alpha, beta, gamma) | `compute_band_powers` | Relative power in each canonical frequency band (sums to 1) | Delta/theta increased, alpha reduced | Theta increased (frontal-dominant) |
| **Spectral: Ratio** | Theta/Alpha ratio | Derived from band powers | Theta power divided by alpha power — captures spectral shift in a single number | Elevated | Elevated |
| **Spectral: Ratio** | Slowing ratio | `compute_slowing_ratio` | (delta + theta) / (alpha + beta) — the single most robust spectral AD marker | Elevated | Elevated |
| **Spectral: Asymmetry** | Frontal/Posterior asymmetry | `compute_frontal_posterior_asymmetry` | Ratio of (theta + alpha) power in frontal vs posterior channels | <1 (posterior-dominant slowing) | >1 (frontal-dominant slowing) |
| **Connectivity: Coherence** | Posterior alpha coherence | `compute_coherence_posterior` | Mean magnitude-squared coherence in 8–13 Hz across posterior electrode pairs | Reduced (posterior network breakdown) | Less affected |
| **Connectivity: Coherence** | Global alpha coherence | `compute_coherence_global` | Mean magnitude-squared coherence in 8–13 Hz across all channel pairs | Moderately reduced | Reduced (diffuse disconnection) |
| **Connectivity: Phase** | Phase Lag Index (PLI) | `compute_pli` | Phase synchronisation in alpha band, robust to volume conduction artifacts | Reduced | Reduced |
| **Complexity** | Lempel-Ziv complexity | `compute_lempel_ziv` | Signal irregularity based on substring repetition in binarised signal | Reduced (more regular/repetitive) | Reduced |
| **Complexity** | Permutation entropy | `compute_permutation_entropy` | Complexity of temporal ordering of amplitude values (ordinal patterns) | Reduced (more predictable) | Reduced |
