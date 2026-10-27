import json, numpy as np
from scipy.optimize import curve_fit
r = json.load(open("results/theory_results.json"))
f = np.array([100, 400, 800, 1000, 2000.])
K = list(r["plt14_freq"])
def inv(f, y0, k): return y0 + k / f
out = {}
for q in ["Pr_plus", "Uc_plus", "loop_area", "fit_Ps"]:
    y = np.array([r["plt14_freq"][k][q] for k in K])
    p, c = curve_fit(inv, f, y); res = y - inv(f, *p)
    # compare with power law
    b = np.polyfit(np.log(f), np.log(y), 1); resp = y - np.exp(np.polyval(b, np.log(f)))
    out[q] = dict(y_inf=p[0], y_inf_err=np.sqrt(c[0, 0]), k=p[1], k_err=np.sqrt(c[1, 1]), rms_1overf=np.sqrt(np.mean(res**2)), rms_powerlaw=np.sqrt(np.mean(resp**2)), exponent=b[0])
    print(f"{q:10s} 1/f: y_inf={p[0]:.2f}±{np.sqrt(c[0,0]):.2f} k={p[1]:.0f}±{np.sqrt(c[1,1]):.0f} rms={out[q]['rms_1overf']:.3f} | power-law exp={b[0]:.3f} rms={out[q]['rms_powerlaw']:.3f}")
# leakage estimate for a triangular bipolar wave (amplitude V0, centred loop): DeltaPr ~ V0/(8 f R A)  -> R*A = V0/(8 k)
V0 = 15.0; k = out["Pr_plus"]["k"] * 1e-2          # uC/cm2*Hz -> C/m2/s
RA = V0 / (8 * k); A = np.pi * (75e-6) ** 2
out["leak"] = dict(RA_ohm_m2=RA, R_ohm=RA / A, J_at_V0_A_cm2=V0 / RA * 1e-4)
print("leakage estimate:", {a: f"{b:.3g}" for a, b in out["leak"].items()})
json.dump(out, open("results/leak_fit.json", "w"), indent=1, default=float)
