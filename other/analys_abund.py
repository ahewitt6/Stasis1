import os

# make the script’s directory the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(script_dir)

import numpy as np

# — hard-coded inputs —
Tfile           = "PRyMrates/thermo/Tgamma_Tnu.txt"
rho_file        = "PRyMrates/thermo/rho_matter_radiation.txt"
E_hi, E_lo = 8.0, 2.0   # MeV stasis window

# — load Tγ vs t —
Tdata = np.loadtxt(Tfile, skiprows=1)
t_T   = Tdata[:,0]    # time [s]
Tg    = Tdata[:,1]    # photon temp [MeV]

# — load ρₘ, ρᵣ vs t —
Rdata = np.loadtxt(rho_file, skiprows=1)
t_R   = Rdata[:,0]
rho_m = Rdata[:,1]
rho_r = Rdata[:,2]

# — find common times —
t_common, idx_T, idx_R = np.intersect1d(t_T, t_R, return_indices=True)

if len(t_common)==0:
    raise RuntimeError("No overlapping times between T-file and rho-file!")

# — slice to common grid —
Tg_c    = Tg[idx_T]
rho_m_c = rho_m[idx_R]
rho_r_c = rho_r[idx_R]

# — compute Omegas —
total   = rho_m_c + rho_r_c
omega_m = rho_m_c / total
omega_g = rho_r_c / total

# — mask stasis window in energy —
mask    = (Tg_c <= E_hi) & (Tg_c >= E_lo)
print(f"\nStasis [{E_lo:.2f}, {E_hi:.2f}] MeV  →  {mask.sum()} points")
print(f"  ⟨Ωₘ⟩ = {omega_m[mask].mean():.3e} ± {omega_m[mask].std():.3e}")
print(f"  ⟨Ωγ⟩ = {omega_g[mask].mean():.3e} ± {omega_g[mask].std():.3e}")

# — print a few sample rows —
print("\nTγ [MeV]    Ωₘ        Ωγ")
for Tg_i, Om, Og in zip(Tg_c[:10], omega_m[:10], omega_g[:10]):
    print(f"{Tg_i:8.3f}   {Om:.3e}   {Og:.3e}")