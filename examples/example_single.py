# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Example how to call the C_ij function from the `PecVelCov.jl` package for a
single pair of points.
"""
import matplotlib.pyplot as plt
import numpy as np
import pecvelcov


def load_Pk(fpath, make_plot):
    ks, Pk = np.load(fpath).T

    if make_plot:
        plt.figure()
        plt.plot(ks, Pk)
        plt.xlabel(r"$r ~ [\mathrm{Mpc} / h]$")
        plt.ylabel(r"$P(k) ~ [(\mathrm{Mpc} / h)^3]$")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()

        fname_out = "./Pk.png"
        print(f"Saving power spectrum plot to `{fname_out}`.")
        plt.savefig(fname_out, dpi=450)
        plt.close()

    return ks, Pk


if __name__ == "__main__":
    ks, Pk = load_Pk("../data/pk_fiducial.npy", True)

    kwargs = {"ri": 20.0, "rj": 30.0, "cos_theta": 0.9,
              "Pk": Pk, "ks": ks, "ell_min": 0, "ell_max": 50}
    print("Calculating C_ij for:")
    for key, val in kwargs.items():
        print(f"{key:<20} {val}")

    print("Calling C_ij...")
    C_ij = pecvelcov.C_ij(**kwargs)
    print(f"C_ij = {C_ij}")
