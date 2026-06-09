#!/usr/bin/env python3
"""
make_inputs_from_psin.py

• Discovers jetto_*.eqdsk_out (and jetto.eqdsk_out)
• For each equilibrium:
      – loops over a user-supplied list of ψ_n values
      – writes a GENE input file per surface
      – plots the flux-surfaces (R vs Z) and saves the figure
"""

from pathlib import Path
import glob
import os
from scipy.interpolate import interp1d


# ───────────── libraries ───────────── #
import matplotlib
matplotlib.rcParams.update({          # ❶ global style
    "font.size":    10,               #    – all text 10 pt
    "text.usetex":  False             #    – no LaTeX renderer
})
import matplotlib.pyplot as plt
import numpy as np
from pyrokinetics import Pyro
# ───────────────────────────────────── #

# ─────── USER SETTINGS ─────── #
#psin_list = [0.10, 0.15, 0.20, 0.25, 0.30,
#             0.35, 0.40, 0.45, 0.50, 0.55,
#             0.60, 0.65, 0.70, 0.75, 0.80,
#             0.85, 0.90]

r_list_t3d = [0.082, 0.245, 0.409, 0.573, 0.736, 0.9]

r_list_t3d_giacomin = [0.16, 0.33, 0.49, 0.65, 0.82, 0.1]

r_list_profiles = np.linspace(0.05, 0.95, 37).tolist()

r_list = r_list_t3d_giacomin

gk_code = "GENE"                # or CGYRO, STELLA, …

Gyro_directory = Path(os.environ["GYRO_DATA_DIR"])
Case = "S1"
eq_glob  = Gyro_directory  / "GENE" / "Templates" / Case / "jetto_*.eqdsk_out"
plain_eq = Gyro_directory / "GENE" / "Templates" / Case / "jetto.eqdsk_out"


out_dir  = Path("inputs"); out_dir.mkdir(exist_ok=True)
plot_dir = Path("Plots");  plot_dir.mkdir(exist_ok=True)
# ──────────────────────────── #

# -------- discover equilibria & sort by time -------- #
eq_files = glob.glob(str(eq_glob))
if Path(plain_eq).exists():
    eq_files.append(plain_eq)

def time_from(fname: str) -> float:
    """float time from jetto_<time>.eqdsk_out or −1 for plain file"""
    base = os.path.basename(fname)
    if base == "jetto.eqdsk_out":
        return -1.0
    return float(base.replace("jetto_", "").replace(".eqdsk_out", ""))

file_info = sorted(((time_from(f), f) for f in eq_files), key=lambda x: x[0])

# ---------------------- main loop ------------------- #
for tval, eq_file in file_info:
    kin_file = Path(eq_file).with_suffix(".jsp")
    if not kin_file.exists():
        print(f"!! missing {kin_file} – skipping {eq_file}")
        continue

    print(f"\n>>> {eq_file}  (t = {tval:+.6f})")

    pyro = Pyro(eq_file=eq_file,
                kinetics_file=str(kin_file),
                eq_type="GEQDSK",
                eq_kwargs={"psi_n_lcfs": 0.999})

    R_surf, Z_surf = {}, {}
    
    psin_list = []

    for r_a in r_list:
        pyro = Pyro(eq_file=eq_file,
                    kinetics_file=str(kin_file),
                    eq_type="GEQDSK",
                    eq_kwargs={"psi_n_lcfs": 0.999})
        # local geometry & GK input
        r_to_psi = interp1d(pyro.eq['rho'], pyro.eq['psi_n'])
        psin = r_to_psi([r_a])[0]
        psin_list.append(psin)
        pyro.load_local(psi_n=psin, local_geometry="Miller")
        pyro.load_local_geometry(psi_n=psin, local_geometry="Miller", show_fit=False)
        pyro.load_local_geometry(psi_n=psin, local_geometry="Miller", show_fit=False)
        outname = out_dir / f"t{tval:+09.3f}_psin{psin:.3f}.in"
        pyro.local_species.merge_species("deuterium",["tritium","alpha","impurity1","impurity2"],keep_base_species_mass=False,keep_base_species_z=False)
        #pyro.write_gk_file(file_name=outname, gk_code=gk_code)

        # store surface for plotting
        surf = pyro.eq.flux_surface(psi_n=psin)
        R_surf[psin] = surf['R'].data.magnitude
        Z_surf[psin] = surf['Z'].data.magnitude

    # ------------------ plot all chosen surfaces ------------------ #
    red_r = {0.65, 0.82}
    plt.figure(figsize=(5.25, 3.75))          # ❷ fixed fig-size
    for r_a, psin in zip(r_list, psin_list):
        plt.plot(R_surf[psin], Z_surf[psin], 'r' if r_a in red_r else 'k')
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.axis('square')

    R_outer = R_surf[max(psin_list)]
    plt.xlim(min(R_outer) * 0.9, max(R_outer) * 1.1)

    plt.tight_layout()
    pfile = plot_dir / f"t{tval:+09.3f}_surfaces.pdf"
    plt.savefig(pfile, bbox_inches='tight', pad_inches=0.05)
    pfile = plot_dir / f"t{tval:+09.3f}_surfaces.png"
    plt.savefig(pfile, bbox_inches='tight', pad_inches=0.05, dpi=150)
    plt.close()

    print(f"    → {len(psin_list)} inputs + {pfile}")

print("\nDone!")

