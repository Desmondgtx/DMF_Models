# DMF_Models

Dynamic Mean Field (DMF) whole-brain model that generates BOLD-like signals from
a structural connectivity matrix, plus a hemodynamic (Balloon–Windkessel) forward
model and a resting-state HRF estimation/deconvolution pipeline. The code simulates
functional connectivity, compares it against empirical group FC, and estimates
hemodynamic response functions from both simulated and empirical fMRI.

## Context

<!-- Confirm collaborators / funding wording with Diego. -->
Computational neuroscience code developed for a FONDECYT-funded project on
whole-brain modeling and hemodynamic response estimation. The DMF and hemodynamic
model implementations derive from prior work by Carlos Coronel and Patricio Orio;
the HRF-estimation and empirical-comparison code is by Diego Garrido. No associated
paper/poster is referenced yet (see Citation).

## What's in this repo

Core model and pipeline (Python):

- `DMF.py` — Dynamic Mean Field model (excitatory/inhibitory mean-field with
  feedback inhibition control). Integrates the network and returns BOLD-like
  signals and firing rates.
- `BOLDModel.py` — Generalized hemodynamic (Balloon–Windkessel) model mapping
  firing rates to BOLD signals (Stephan et al. 2007; Deco et al. 2018).
- `RunDMF.py` — Main pipeline: loads structural connectivity, runs a DMF
  simulation, filters the BOLD, computes FC/FCD, and estimates HRFs.
- `deconv.py` — Wrapper around the `rsHRF` library for resting-state HRF
  estimation and Wiener deconvolution of BOLD time series.
- `deconv_Wilcoxon_VM.py` — ROI-level HRF estimation on empirical fMRI (nilearn)
  with a Wilcoxon comparison between conditions.
- `boldImpulse.py` — Minimal BOLD impulse-response demonstration.

Inputs and reference data:

- `structural_Deco_AAL.txt`, `SC_opti_25julio.txt` — structural connectivity
  matrices (AAL-90 parcellation).
- `average_90x90FC_HCPchina_symm.npy` — empirical group functional connectivity
  used as the fitting/comparison target.

Data folders (see **Data** — human-subject data, review before sharing):

- `Subjects/` — empirical BOLD time series, 22 subjects.
- `Subjects MATLAB/` — per-subject HRF estimates (`.mat`).
- `Subjects Medel/` — preprocessed fMRI volumes (NIfTI) plus confound regressors.

Supporting:

- `Demo/` — `rsHRF` demo scripts and example surface/volume data.
- `Scripts exploratorios/` — exploratory / development scripts (not part of the
  main pipeline).
- `Scripts originales/` — original reference versions of the model scripts.
- `Presentacion/` — figures.

## How to reproduce

Requires Python 3.9+. Install dependencies (`anarpy` is fetched from GitHub):

```bash
pip install numpy scipy matplotlib numba rsHRF nilearn nibabel
pip install git+https://github.com/vandal-uv/anarpy.git
```

Run the main DMF simulation and FC/HRF analysis:

```bash
python RunDMF.py
```

Run the model with a random structural matrix (no external data needed):

```bash
python DMF.py
```

Estimate HRFs from empirical fMRI (nilearn/rsHRF):

```bash
python deconv_Wilcoxon_VM.py   # NOTE: edit the hardcoded input paths first
```

`RunDMF.py` expects `SC_opti_25julio.txt` and `average_90x90FC_HCPchina_symm.npy`
in the working directory (both included).

## Data

- Structural connectivity matrices and the group FC target are included and
  sufficient to run the simulation pipeline.
- The `Subjects/`, `Subjects MATLAB/`, and `Subjects Medel/` folders contain
  human-subject neuroimaging data. Review consent, anonymization, and data-sharing
  terms before distributing this repository; the simulation pipeline
  (`RunDMF.py`, `DMF.py`) does not require them.

## Citation

<!-- No paper/poster confirmed yet. Add BibTeX here once available. -->

## License

<!-- No LICENSE file yet. Recommended: MIT for the analysis code. -->
