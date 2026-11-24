# ShowSimulations

Streamlit app to inspect the **timing of MRzero (simulated) sequences**, in
particular cardiac-triggered, segmented inversion-recovery (LGE) sequences
played with irregular heart rhythms: when each k-space line is acquired, which
recovery duration (time between consecutive inversions, i.e. RR interval) it
sees, and the resulting longitudinal magnetization.

It is the simulation counterpart of [ShowTwix](https://github.com/CyprienBouton/ShowTwix),
which shows the same information for Siemens Twix raw data acquired on the scanner.
It does not read raw signals or reconstruct images: only the sequence is needed.

## Installation

```
git clone https://github.com/CyprienBouton/ShowSimulations.git
cd ShowSimulations
conda create -n ShowSimulations python=3.10
conda activate ShowSimulations
pip install -r requirements.txt
```

Main dependencies: `streamlit`, `mrzerocore` (pulls `torch`), `pandas`, `plotly`, `scipy`.

## Usage

```
streamlit run app.py
```

The app opens at http://localhost:8501. Uploads are limited to 1000 MB
(`.streamlit/config.toml`).

## Expected input

The **played sequence** of a simulation, i.e. the sequence with the actual
(irregular) trigger delays inserted, as produced by `run_simulation_cluster`
(`mrsim`):

* `.pkl`: pickled `MRzeroCore.Sequence`, or
* `.seq`: Pulseq file (imported with `MRzeroCore.Sequence.import_file`).

Inversion pulses are detected as the repetitions with a flip angle close to
180 degrees; k-space positions come from `seq.get_kspace()`.

## Pages

| Page | Content |
| --- | --- |
| **Choose raw data** | Upload the `.pkl` / `.seq` sequence. Builds the per-line table (line, partition, acquisition time, recovery duration). Must be done first. |
| **Acquisition Timeline** | k-space map (line x partition) coloured by acquisition time. |
| **Recovery Durations** | k-space map coloured by the recovery duration of each line (ms), adjustable colour range. |
| **Recovery Durations Stats** | Histogram of the recovery durations. |
| **Longitudinal Magnetizations** | Mz(t) for user-defined T1 species (e.g. myocardium, scar, blood) using the TI, flip angle, triggers and readout times of the sequence; options: SPPRESS, centric/linear reordering, correction pulse (custom `t_a` / `alpha_b`, or one optimized pulse). |

## Code layout

* `app.py`: page selector.
* `streamlit_pages/`: one module per page.
* `utils/RecoMRzero.py`: Cartesian gridding of the sequence k-space and per-line timings
  (trimmed copy of `MRzeroReconstructCoreLib.RecoMRzero`).
* `utils/correct_mrzero_seq.py`: `get_IR`, `get_TI_from_seq` (old copy of
  `run_simulation_cluster/src/correct_mrzero_seq.py`).
* `utils/optimized_pulse.py`: Mz model and correction-pulse optimisation (same file as in ShowTwix).
