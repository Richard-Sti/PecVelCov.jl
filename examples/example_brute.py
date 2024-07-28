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
import numpy as np
import pecvelcov

if __name__ == "__main__":
    fname_Pk = "../data/pk_fiducial.npy"

    # Some example points, the distribution doesn't matter.
    # Lengths are in Mpc/h.
    npoints = 10
    rs = np.random.uniform(0, 100, npoints)
    theta = np.arccos(np.random.uniform(-1, 1, npoints))
    phi = np.random.uniform(0, 2*np.pi, npoints)

    print(f"Running brute-force calculation for {npoints} points...")
    covmat_brute = pecvelcov.CovmatBrute(fname_Pk)
    C = covmat_brute(rs, theta, phi)

    print("Finished! Computed the covariance matrix to be:")
    print(C)
