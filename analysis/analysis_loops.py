"""Theory-based analysis of the digitized P-V loops and XRD data of sol-gel PLTx films (thesis, chap. 3)."""
import json, numpy as np
from scipy.optimize import curve_fit
D = "data/pv_loops/"
pe = np.load(D + "pe.npy", allow_pickle=True).item()
fq = np.load(D + "plt14_loops.npy", allow_pickle=True).item()

def branches(V, P):
    """split a loop into ascending (V increasing) and descending branches"""
    dV = np.gradient(V)
    up, dn = dV > 0, dV < 0
    return (V[up], P[up]), (V[dn], P[dn])

def miller(V, Ps, Vc, d, C, sgn):
    return Ps * np.tanh((V - sgn * Vc) / (2 * d)) + C * V

def fit_loop(V, P):
    (Vu, Pu), (Vd, Pd) = branches(V, P)
    X = np.concatenate([Vu, Vd]); Y = np.concatenate([Pu, Pd]); S = np.concatenate([np.ones_like(Vu), -np.ones_like(Vd)])
    f = lambda xs, Ps, Vc, d, C: miller(xs[0], Ps, Vc, d, C, xs[1])
    p0 = [max(abs(Y)) * 0.6, 3, 1.5, 0.3]
    p, cov = curve_fit(f, np.vstack([X, S]), Y, p0=p0, bounds=([0, 0, 0.05, -5], [100, 30, 30, 10]), maxfev=20000)
    res = Y - f(np.vstack([X, S]), *p)
    return dict(Ps=p[0], Vc=p[1], delta=p[2], C=p[3], rms=float(np.sqrt(np.mean(res**2))), R2=1 - np.var(res) / np.var(Y))

def loop_metrics(V, P):
    """Pr, Uc from zero crossings; energies per area (uJ/cm2 = V*uC/cm2) from the positive half (charging = branch from 0 to Vmax)."""
    s = np.where(np.diff(np.sign(V)) != 0)[0]
    Pr = [P[i] + (P[i+1]-P[i]) * (-V[i]) / (V[i+1]-V[i]) for i in s]
    s2 = np.where(np.diff(np.sign(P)) != 0)[0]
    Uc = [V[i] + (V[i+1]-V[i]) * (-P[i]) / (P[i+1]-P[i]) for i in s2]
    # loop area (hysteresis loss per cycle, uJ/cm2) via shoelace
    A = 0.5 * abs(np.dot(V, np.roll(P, 1)) - np.dot(P, np.roll(V, 1)))
    # recoverable energy on the discharge branch in the first quadrant: W = int_{Pr}^{Pmax} V dP
    (Vu, Pu), (Vd, Pd) = branches(V, P)
    m = (Vd > 0); vd, pd = Vd[m], Pd[m]; o = np.argsort(pd)
    Wrec = np.trapezoid(vd[o], pd[o])
    m = (Vu > 0); vu, pu = Vu[m], Pu[m]; o = np.argsort(pu)
    Wtot = np.trapezoid(vu[o], pu[o])
    return dict(Pr_plus=max(Pr), Pr_minus=min(Pr), Uc_plus=max(Uc), Uc_minus=min(Uc), Pmax=P.max(), Vmax=V.max(),
                loop_area=A, W_rec=Wrec, W_charge=Wtot, eta=Wrec / Wtot if Wtot > 0 else np.nan)

out = {"loops_2kHz": {}, "plt14_freq": {}}
for k, v in pe.items():
    V, P = np.asarray(v); m = loop_metrics(V, P); m.update({f"fit_{a}": b for a, b in fit_loop(V, P).items()})
    out["loops_2kHz"][k] = m
for k in ["100 Hz", "400 Hz", "800 Hz", "1000 Hz", "2000 Hz"]:
    V, P = np.asarray(fq[k]); m = loop_metrics(V, P); m.update({f"fit_{a}": b for a, b in fit_loop(V, P).items()})
    out["plt14_freq"][k] = m

# frequency scaling Uc ~ f^beta, area ~ f^alpha
f = np.array([100, 400, 800, 1000, 2000.])
Uc = np.array([out["plt14_freq"][k]["Uc_plus"] for k in out["plt14_freq"]])
Ucm = np.array([-out["plt14_freq"][k]["Uc_minus"] for k in out["plt14_freq"]])
A = np.array([out["plt14_freq"][k]["loop_area"] for k in out["plt14_freq"]])
Pr = np.array([out["plt14_freq"][k]["Pr_plus"] for k in out["plt14_freq"]])
def pl(x, y):
    p, cov = np.polyfit(np.log(x), np.log(y), 1, cov=True); return p[0], np.sqrt(cov[0, 0])
out["scaling"] = {"beta_Uc_plus": pl(f, Uc), "beta_Uc_mean": pl(f, (Uc + Ucm) / 2), "alpha_area": pl(f, A), "gamma_Pr": pl(f, Pr),
                  "beta_Uc_plus_f>=400": pl(f[1:], Uc[1:]), "imprint_V": list((Uc - Ucm) / 2)}

# electrostriction: spontaneous strain x3 - x1 = (Q11 - Q12) Ps^2 (tetragonal), PbTiO3 coefficients (Haun et al. 1987)
Q11, Q12 = 0.089, -0.026
xrd = json.load(open("data/xrd/xrd_results.json"))
est = {}
for s in xrd:
    if s.get("ca", 1) > 1.001:
        Ps = np.sqrt((s["ca"] - 1) / (Q11 - Q12)) * 100       # C/m2 -> uC/cm2
        est[f"{s['sample']}/{s['substrate']}"] = dict(ca=s["ca"], Ps_uCcm2=Ps)
    else:   # upper bound for undetected splitting: (c-a)/a <= FWHM(100)/2 * cot(theta)
        pk = min(s["peaks"], key=lambda p: abs(p["tt"] - 22.5))
        th = np.radians(pk["tt"] / 2); dca = np.radians(pk["fwhm"]) / 2 / np.tan(th)
        est[f"{s['sample']}/{s['substrate']}"] = dict(ca_max=1 + dca, Ps_max_uCcm2=np.sqrt(dca / (Q11 - Q12)) * 100, peak=pk["tt"], fwhm=pk["fwhm"])
out["electrostriction"] = est
json.dump(out, open("results/theory_results.json", "w"), indent=1, default=float)

print("2 kHz loops:")
for k, m in out["loops_2kHz"].items():
    print(f" {k:6s} Pr={m['Pr_plus']:.2f}/{m['Pr_minus']:.2f}  Uc={m['Uc_plus']:.2f}/{m['Uc_minus']:.2f}  Vmax={m['Vmax']:.1f}  Pmax={m['Pmax']:.1f}  "
          f"Wrec={m['W_rec']:.1f} Wch={m['W_charge']:.1f} eta={m['eta']:.2f} area={m['loop_area']:.0f} | fit Ps={m['fit_Ps']:.1f} Vc={m['fit_Vc']:.2f} d={m['fit_delta']:.2f} C={m['fit_C']:.3f} R2={m['fit_R2']:.3f}")
print("PLT14 vs f:")
for k, m in out["plt14_freq"].items():
    print(f" {k:8s} Pr={m['Pr_plus']:.2f} Uc+={m['Uc_plus']:.2f} Uc-={m['Uc_minus']:.2f} area={m['loop_area']:.0f} fitPs={m['fit_Ps']:.1f} Vc={m['fit_Vc']:.2f} C={m['fit_C']:.3f} R2={m['fit_R2']:.3f}")
print("scaling:", {k: (np.round(v, 3) if not isinstance(v, list) else np.round(v, 2)) for k, v in out["scaling"].items()})
print("electrostriction:")
for k, v in est.items(): print(" ", k, {a: round(b, 3) for a, b in v.items()})
