import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analysis_loops import pe, fit_loop, miller, branches   # re-runs the analysis (fast)
r = json.load(open("results/theory_results.json")); L = json.load(open("results/leak_fit.json"))
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
C = {"PT": "k", "PLT7": "#1f77b4", "PLT14": "#2ca02c", "PLT21": "#d62728"}
for k, v in pe.items():
    V, P = np.asarray(v); ax[0].plot(V, P, color=C[k], lw=1.2, label=k)
    p = fit_loop(V, P); vv = np.linspace(V.min(), V.max(), 200)
    for s in (1, -1): ax[0].plot(vv, miller(vv, p["Ps"], p["Vc"], p["delta"], p["C"], s), color=C[k], lw=0.6, ls="--")
ax[0].set_xlabel("V (V)"); ax[0].set_ylabel(r"P ($\mu$C/cm$^2$)"); ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("(a) 2 kHz loops and fits", fontsize=8.5)
f = np.array([100, 400, 800, 1000, 2000.]); ff = np.linspace(90, 2100, 200); K = list(r["plt14_freq"])
for q, col, lab in [("Pr_plus", "#2ca02c", r"$P_r$ ($\mu$C/cm$^2$)"), ("Uc_plus", "#9467bd", r"$+U_c$ (V)")]:
    y = np.array([r["plt14_freq"][k][q] for k in K]); ax[1].plot(f, y, "o", color=col, ms=4, label=lab)
    ax[1].plot(ff, L[q]["y_inf"] + L[q]["k"] / ff, "-", color=col, lw=1)
    ax[1].plot(ff, np.exp(np.polyval(np.polyfit(np.log(f), np.log(y), 1), np.log(ff))), ":", color=col, lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel("f (Hz)"); ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title(r"(b) PLT14: $y_\infty + k/f$ (—) vs $f^\beta$ (···)", fontsize=8)
e = r["electrostriction"]; xs = {"PT": 0, "PLT7": 7, "PLT14": 14, "PLT21": 21}
for k, x in xs.items():
    d = e[f"{k}/LNO"]
    if "Ps_uCcm2" in d: ax[2].plot(x, d["Ps_uCcm2"], "s", color="k", ms=5)
    else: ax[2].errorbar(x, d["Ps_max_uCcm2"] / 2, yerr=d["Ps_max_uCcm2"] / 2, fmt="none", ecolor="gray", capsize=3)
    m = r["loops_2kHz"][k]; ax[2].plot(x, m["fit_Ps"], "o", color=C[k], ms=5); ax[2].plot(x, m["Pr_plus"], "^", color=C[k], mfc="none", ms=5)
ax[2].plot([], [], "s", color="k", label=r"$P_s$ from $c/a$"); ax[2].plot([], [], "-", color="gray", label=r"$P_s$ bound (XRD)")
ax[2].plot([], [], "o", color="0.4", label=r"switchable $P_s$ (fit)"); ax[2].plot([], [], "^", color="0.4", mfc="none", label=r"$P_r$")
ax[2].set_xlabel("La content (mol%)"); ax[2].set_ylabel(r"$\mu$C/cm$^2$"); ax[2].legend(fontsize=6, frameon=False); ax[2].set_title("(c) structure vs switching", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/fig_theory.pdf"); fig.savefig("figures/fig_theory.png", dpi=200)
