import json
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# PLT step 1 (Kaggle GPU) — La-substituted PbTiO3 supercells with universal ML potentials
Pb1-xLaxTi1-x/4O3 (B-site vacancy compensation, nominal composition of the sol-gel films) and Pb1-3x/2Lax(V)x/2TiO3 (A-site vacancies),
x = 0 to 0.3125 in 4x4x4 supercells (about 320 atoms), three random configurations each, relaxed at 0 K with two models
(MACE-MatPES-r2SCAN-0 and MACE-MPA-0), then NPT molecular dynamics at 300 K for the B-site series.

**Run on Kaggle:** Settings → Accelerator **GPU T4 x2**, Internet **On** → Save Version → **Save & Run All (Commit)**.
Outputs (Output tab, written progressively): `plt_relax.csv`, `plt_md300.csv`, `plt_structures.extxyz`. Stops cleanly after 11 h.""")

code("""# 1. Install
!pip install -q mace-torch spglib 2>&1 | tail -1
import importlib.metadata as im, torch
print("mace-torch", im.version("mace-torch"), "| cuda", torch.cuda.is_available())
""")

code("""# 2. Setup
import os, time, numpy as np, pandas as pd
from ase import Atoms, units
from ase.io import write
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.neighborlist import neighbor_list
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from mace.calculators import mace_mp
W = "/kaggle/working"; T_START = time.time(); BUDGET = 11.0 * 3600
MODELS = ["mace-matpes-r2scan-0", "medium-mpa-0"]
N = 4                                    # 4x4x4 supercell: 64 Pb sites, 64 Ti sites
NLA = [0, 4, 8, 12, 16, 20]              # x = 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125
SEEDS = [0, 1, 2]
COMP = ["B", "A"]                        # B: one Ti vacancy per 4 La (nominal); A: one Pb vacancy per 2 La

def unit(a=3.9):
    return Atoms(["Pb", "Ti", "O", "O", "O"], scaled_positions=[[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]], cell=np.eye(3)*a, pbc=True)

def relax_unit(calc_unused=None, model=None):
    calc = mace_mp(model=model, device="cuda", default_dtype="float64")     # unit cell in double precision
    at = unit(); p = at.get_positions(); p[1, 2] += 0.15; p[2:, 2] -= 0.08; at.set_positions(p); at.calc = calc
    at.set_constraint(FixSymmetry(at, symprec=1e-4)); BFGS(FrechetCellFilter(at), logfile=None).run(fmax=1e-3, steps=2000); at.set_constraint()
    at.calc = None; return at

# Ti O6 cages from the ideal cubic supercell (same atom order as all doped cells before deletions)
ideal = unit(4.0).repeat((N, N, N)); sym0 = np.array(ideal.get_chemical_symbols())
i, j = neighbor_list("ij", ideal, 2.2)
CAGE = {b: [jj for ii, jj in zip(i, j) if ii == b and sym0[jj] == "O"] for b in np.where(sym0 == "Ti")[0]}
assert all(len(v) == 6 for v in CAGE.values())
PB = np.where(sym0 == "Pb")[0]; TI = np.where(sym0 == "Ti")[0]

def build(polar_uc, nla, comp, seed):
    \"\"\"returns doped supercell and the index map old->new (None for deleted atoms)\"\"\"
    rng = np.random.default_rng(1000 * nla + 17 * seed + (0 if comp == "B" else 7))
    at = polar_uc.repeat((N, N, N)); s = np.array(at.get_chemical_symbols())
    la = rng.choice(PB, nla, replace=False); s[la] = "La"
    if comp == "B": dele = rng.choice(TI, nla // 4, replace=False)
    else:           dele = rng.choice(np.setdiff1d(PB, la), nla // 2, replace=False)
    at.set_chemical_symbols(list(s))
    keep = np.setdiff1d(np.arange(len(at)), dele); new = {o: k for k, o in enumerate(keep)}
    return at[keep], new, len(dele)

def metrics(at, new):
    pos = at.get_positions(); cell = at.cell[:]; inv = np.linalg.inv(cell); d = []
    for b, cg in CAGE.items():
        if b not in new: continue
        v = pos[[new[o] for o in cg]] - pos[new[b]]; v -= np.round(v @ inv) @ cell; d.append(-v.mean(0))
    d = np.array(d); L = np.sort(at.cell.lengths() / N)
    return dict(a1=L[0], a2=L[1], a3=L[2], ca=L[2] / L[0], apc=(at.get_volume() / N**3) ** (1/3),
                dTi_mean_vec=np.linalg.norm(d.mean(0)), dTi_local=np.linalg.norm(d, axis=1).mean())

def save(rows, fn):
    pd.DataFrame(rows).to_csv(f"{W}/{fn}", index=False)
print("supercell atoms:", len(ideal))
""")

code("""# 3. 0 K relaxations (positions + cell, no symmetry constraint), starting from the polar tetragonal supercell
rows = []
for model in MODELS:
    calc = mace_mp(model=model, device="cuda", default_dtype="float32")
    uc = relax_unit(model=model); cp = uc.cell.cellpar()
    print(f"{model}: PbTiO3 unit cell c/a = {cp[2]/cp[0]:.4f}", flush=True)
    E_ref = {}
    for comp in COMP:
        for nla in NLA:
            if comp == "A" and nla == 0: continue
            for seed in SEEDS:
                if nla == 0 and seed > 0: continue
                if time.time() - T_START > BUDGET: break
                t0 = time.time()
                at, new, nv = build(uc, nla, comp, seed); at.calc = calc
                opt = FIRE(FrechetCellFilter(at), logfile=None); conv = opt.run(fmax=0.02, steps=1500)
                m = metrics(at, new)
                rows.append(dict(model=model, comp=comp, n_La=nla, x=nla / 64, n_vac=nv, seed=seed, natoms=len(at),
                                 E_eV=at.get_potential_energy(), converged=bool(conv), steps=opt.nsteps, time_s=round(time.time() - t0), **m))
                save(rows, "plt_relax.csv")
                if seed == 0:
                    a2 = at.copy(); a2.calc = None; a2.info.update(model=model, comp=comp, n_La=nla); write(f"{W}/plt_structures.extxyz", a2, append=True)
                print(f"{model[:14]:14s} {comp} x={nla/64:.3f} seed {seed}: c/a={m['ca']:.4f} a_pc={m['apc']:.4f} |<dTi>|={m['dTi_mean_vec']:.3f} "
                      f"<|dTi|>={m['dTi_local']:.3f} conv={conv} {opt.nsteps} steps {time.time()-t0:.0f}s", flush=True)
    del calc; torch.cuda.empty_cache()
""")

code("""# 4. NPT molecular dynamics at 300 K (B-site series, seed 0): does the polar state survive at room temperature?
md_rows = []; DT, NSTEP, NAVG = 2.0, 2500, 1250
for model in MODELS:
    calc = mace_mp(model=model, device="cuda", default_dtype="float32"); uc = relax_unit(model=model)
    for nla in NLA:
        if time.time() - T_START > BUDGET: print("time budget reached"); break
        t0 = time.time(); at, new, nv = build(uc, nla, "B", 0); at.calc = calc
        FIRE(FrechetCellFilter(at), logfile=None).run(fmax=0.05, steps=500)
        MaxwellBoltzmannDistribution(at, temperature_K=300); Stationary(at)
        dyn = Inhomogeneous_NPTBerendsen(at, timestep=DT*units.fs, temperature_K=300, pressure_au=0.0,
                                         taut=100*units.fs, taup=1000*units.fs, compressibility_au=4.57e-5/units.bar)
        rec = []
        def sample():
            if dyn.nsteps >= NSTEP - NAVG and dyn.nsteps % 10 == 0:
                m = metrics(at, new); rec.append([m["ca"], m["apc"], m["dTi_mean_vec"], m["dTi_local"], at.get_temperature()])
        dyn.attach(sample, interval=1); dyn.run(NSTEP)
        r = np.array(rec)
        md_rows.append(dict(model=model, comp="B", n_La=nla, x=nla/64, T=300, ca=r[:,0].mean(), ca_std=r[:,0].std(), apc=r[:,1].mean(),
                            dTi_mean_vec=r[:,2].mean(), dTi_mean_vec_std=r[:,2].std(), dTi_local=r[:,3].mean(), T_md=r[:,4].mean(), time_s=round(time.time()-t0)))
        save(md_rows, "plt_md300.csv")
        print(f"MD300 {model[:14]:14s} x={nla/64:.3f}: c/a={r[:,0].mean():.4f} |<dTi>|={r[:,2].mean():.3f} <|dTi|>={r[:,3].mean():.3f} ({(time.time()-t0)/NSTEP:.3f} s/step)", flush=True)
    del calc; torch.cuda.empty_cache()
print("done", f"{(time.time()-T_START)/3600:.1f} h")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("/home/claude/these/theory/plt_step1_supercells_kaggle.ipynb", "w"), indent=1)
print("ok")
