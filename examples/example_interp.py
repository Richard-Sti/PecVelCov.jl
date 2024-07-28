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

    Cii_path = "/mnt/extraspace/rstiskalek/BBF/Cii_grid.jld2"
    Cij_path = "/mnt/extraspace/rstiskalek/BBF/Cij_grid.jld2"
    Cij_close_path = "/mnt/extraspace/rstiskalek/BBF/Cij_close_grid.jld2"
    Cij_opposite_path = "/mnt/extraspace/rstiskalek/BBF/Cij_opposite_grid.jld2"

    # Some example points, the distribution doesn't matter.
    # Lengths are in Mpc/h.
    npoints = 2500
    rs = np.random.uniform(0, 100, npoints)
    theta = np.arccos(np.random.uniform(-1, 1, npoints))
    phi = np.random.uniform(0, 2*np.pi, npoints)

    print(f"Running fast calculation for {npoints} points...")
    covmat_builder = pecvelcov.CovmatInterp(
        Cii_path, Cij_path, Cij_close_path, Cij_opposite_path)
    C = covmat_builder(rs, theta, phi)

    print("Finished! Computed the covariance matrix to be:")
    print(C)
    print("Covariance matrix shape:", C.shape)
