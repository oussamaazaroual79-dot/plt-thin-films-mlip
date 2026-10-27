# La-rich sol–gel PLT thin films: data, loop analysis and machine-learning molecular dynamics

Data and code for: O. Azaroual, *Hidden tetragonality and collapse of ferroelectricity in La-rich sol–gel Pb1−xLaxTi1−x/4O3 thin films: experiment, hysteresis-loop analysis and machine-learning molecular dynamics* (submitted).

## Contents
- `data/xrd/`: XRD patterns of PT and PLTx films (x = 0.07–0.28) on glass, SiO2/Si and LaNiO3/SiO2/Si (`xrd_<sample>_<substrate>.csv`: 2θ in degrees, normalised intensity). The folder also contains the detected peaks with their FWHM and Scherrer sizes (`xrd_peaks.csv`), the lattice parameters (`xrd_lattice_summary.csv`) and the full results (`xrd_results.json`).
- `data/pv_loops/`: polarisation–voltage loops (V in volts, P in µC/cm²).
  - `loop_2kHz_<film>.csv`: PT, PLT7, PLT14 and PLT21 at 2 kHz.
  - `loop_PLT14_<f>.csv`: PLT14 at 100–2000 Hz, and at 10 V and 15 V amplitude.
  - `pe.npy` and `plt14_loops.npy`: the same loops in the format read by the analysis scripts.
- `data/published/`: frequency series of the PT and PLT7 films as published in El Hasnaoui & Azaroual, ISyDMA'6, Springer 2022, https://doi.org/10.1007/978-3-031-11397-0_7.
- `data/simulations/`: results of the MACE simulations of Pb1−xLaxTi1−x/4O3 in 4×4×4 supercells (x = 0–0.3125), run with the MACE-MatPES-r2SCAN-0 and MACE-MPA-0 models.
  - `plt_relax.csv`: 0 K relaxations, with B-site and A-site vacancy compensation and three random configurations per composition.
  - `plt_md300.csv`: NPT molecular dynamics at 300 K (c/a, pseudocubic parameter, long-range and local Ti off-centring).
  - `plt_structures.extxyz`: relaxed structures.
- `notebooks/`: the Kaggle notebook used for the simulations (`plt_step1_supercells_kaggle.ipynb`, run on a T4 GPU) and the script that generates it.
- `analysis/`: run these from the repository root, in this order:
  - `analysis_loops.py`: tanh (Miller) fits, Pr, Uc, energies and electrostriction bounds → `results/theory_results.json`;
  - `leak_fit.py`: y∞ + k/f and power-law fits of the frequency dependence, leakage estimate → `results/leak_fit.json`;
  - `fig_theory.py` and `fig_mlip.py`: figures 7 and 8.
- `figures/`: figures of the paper.

## Note on the experimental data
The XRD patterns and P–V loops were extracted numerically from the vector figures of the author's PhD thesis (Faculté des Sciences Dhar El Mehraz, Fès). The original instrument files are not available. Peak positions and widths are therefore limited by the resolution of the plotted curves.

## Requirements
- Analysis: Python 3 with numpy, scipy, pandas and matplotlib.
- Simulations: Kaggle GPU session with mace-torch, ase and spglib (installed in the notebook).

## License
Data: CC BY 4.0. Code: MIT.
