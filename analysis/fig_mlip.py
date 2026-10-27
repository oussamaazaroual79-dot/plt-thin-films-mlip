import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
md = pd.read_csv("data/simulations/plt_md300.csv"); rx = pd.read_csv("data/simulations/plt_relax.csv")
th = json.load(open("results/theory_results.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for mod, col, lab in [("mace-matpes-r2scan-0", "#1f77b4", "r$^2$SCAN model, MD 300 K"), ("medium-mpa-0", "#d62728", "PBE model (MPA-0), MD 300 K")]:
    g = md[md.model == mod]; ax[0].plot(100 * g.x, g.ca, "o-", color=col, ms=4, label=lab)
    r = rx[(rx.model == mod) & (rx.comp == "B")].groupby("x").ca.agg(["mean", "std"]).reset_index()
    ax[0].errorbar(100 * r.x, r["mean"], yerr=r["std"], fmt="s", mfc="none", color=col, ms=4, alpha=0.6, label=lab.split(",")[0] + ", 0 K")
e = th["electrostriction"]
ax[0].plot(0, e["PT/LNO"]["ca"], "k*", ms=9, label="films, XRD (this work)")
for k, x in [("PLT7", 7), ("PLT14", 14), ("PLT21", 21)]:
    ax[0].plot([x, x], [1, e[f"{k}/LNO"]["ca_max"]], "k-", lw=2, alpha=0.5)
ax[0].plot([], [], "k-", lw=2, alpha=0.5, label="XRD resolution bound")
ax[0].axhline(1.065, color="gray", ls=":", lw=0.8); ax[0].text(25, 1.07, "bulk PbTiO$_3$ (exp.)", fontsize=6.5, color="gray")
ax[0].set_ylim(0.99, 1.3); ax[0].set_xlabel("La content (mol%)"); ax[0].set_ylabel("$c/a$"); ax[0].legend(fontsize=5.8, frameon=False, loc="upper right")
ax[0].set_title("(a) tetragonality", fontsize=8.5)
g = md[md.model == "mace-matpes-r2scan-0"]
ax[1].plot(100 * g.x, g.dTi_mean_vec / g.dTi_mean_vec.iloc[0], "o-", color="#1f77b4", ms=4, label=r"MD: $|\langle d_{Ti}\rangle|/|\langle d_{Ti}\rangle|_{x=0}$")
ax[1].plot(100 * g.x, g.dTi_local / g.dTi_mean_vec.iloc[0], "o--", color="#1f77b4", mfc="none", ms=4, label=r"MD: local $\langle|d_{Ti}|\rangle$ (same scale)")
L = th["loops_2kHz"]; xs = {"PT": 0, "PLT7": 7, "PLT14": 14, "PLT21": 21}
ax[1].plot([xs[k] for k in xs], [L[k]["Pr_plus"] / L["PT"]["Pr_plus"] for k in xs], "k^", ms=6, label=r"films: $P_r/P_r$(PT), 2 kHz")
ax[1].annotate("PLT21: loop at ±5 V", (21, L["PLT21"]["Pr_plus"] / L["PT"]["Pr_plus"]), (13, 0.12), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
ax[1].set_xlabel("La content (mol%)"); ax[1].set_ylabel("relative polarisation"); ax[1].set_ylim(0, 1.1); ax[1].legend(fontsize=6, frameon=False)
ax[1].set_title("(b) polar order at 300 K", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/fig_mlip.pdf"); fig.savefig("figures/fig_mlip.png", dpi=200)
