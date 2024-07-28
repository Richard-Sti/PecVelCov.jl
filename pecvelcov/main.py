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
Various Python wrappers to the `PecVelCov.jl` package.
"""
from julia import Main
import numpy as np


def C_ij(ri, rj, cos_theta, Pk, ks, ell_min=0, ell_max=50):
    """
    Compute the radial peculiar velocity covariance matrix between two
    galaxies at positions `ri` and `rj`.

    Parameters
    ----------
    ri, rj, cos_theta : float
        Radial positions and cosine of the angle between the two galaxies.
    Pk : 1-dimensional array
        Power spectrum in Mpc^3/h^3.
    ks : 1-dimensional array
        Wavenumbers in h/Mpc.
    ell_min, ell_max : int, optional
        Minimum and maximum multipole moments.

    Returns
    -------
    float
    """
    Main.cos_theta = cos_theta
    Main.ell_min = ell_min
    Main.ell_max = ell_max

    Main.Pells = Main.eval(
        "precompute_legendre_Pells(ell_min, ell_max, cos_theta)")

    Main.kri = np.asarray(ks * ri, dtype=float)
    Main.krj = np.asarray(ks * rj, dtype=float)
    Main.Pk = np.asarray(Pk, dtype=float)
    Main.ks = np.asarray(ks, dtype=float)

    return Main.eval("C_ij(kri, krj, Pells, Pk, ks)")


###############################################################################
#           Brute-force calculation of the covariance matrix                  #
###############################################################################


class CovmatBrute:
    """
    Brute-force calculation of radial peculiar velocity covariance matrix.

    Parameters
    ----------
    fname_Pk : str
        Path to the power spectrum file.
    Pk_npoints : int, optional
        Number of points to evaluate the power spectrum at.
    Pk_transition : float, optional
        Transition scale for the power spectrum below which the power spectrum
        is logarithmically spaced. Above this scale, the power spectrum is
        linearly spaced.
    Pk_log_fraction : float, optional
        Fraction of the power spectrum that is logarithmically spaced.
    """

    def __init__(self, fname_Pk, Pk_npoints=2048, Pk_transition=0.33,
                 Pk_log_fraction=0.33):
        # Load the power spectrum.
        Main.npoints = Pk_npoints
        Main.transition = Pk_transition
        Main.log_fraction = Pk_log_fraction
        Main.fname_Pk = fname_Pk
        Main.eval("ks, Pk = build_Pk(fname_Pk, npoints, transition; log_fraction=log_fraction)")  # noqa

    def __call__(self, rs, theta, phi, ell_min=0, ell_max=50):
        """
        Compute the radial peculiar velocity covariance matrix.

        Parameters
        ----------
        rs : 1-dimensional array
            Radial positions of the galaxies in Mpc/h.
        theta : 1-dimensional array
            Polar angles of the galaxies in radians.
        phi : 1-dimensional array
            Azimuthal angles of the galaxies in radians.
        ell_min : int, optional
            Minimum multipole moment.
        ell_max : int, optional
            Maximum multipole moment.

        Returns
        -------
        2-dimensional array
        """
        # Move the data to Julia
        Main.rs = np.asarray(rs, dtype=float)
        Main.theta = np.asarray(theta, dtype=float)
        Main.phi = np.asarray(phi, dtype=float)

        Main.ell_min = ell_min
        Main.ell_max = ell_max

        return Main.eval(
            "pecvel_covmat_brute(rs, theta, phi, Pk, ks; ell_min=ell_min)")


###############################################################################
#        Fast calculation of the covariance matrix from interpolators         #
###############################################################################


class CovmatInterp:

    def __init__(self, Cii_path, Cij_path, Cij_close_path, Cij_opposite_path,
                 verbose=True):
        if verbose:
            print("Building interpolators.", flush=True)

        Main.eval(f"""
Cii_interp = build_Cii_interpolator("{Cii_path}")
Cij_interp = build_Cij_joint_interpolator(
    "{Cij_path}",
    "{Cij_close_path}",
    "{Cij_opposite_path}")
""")

        if verbose:
            print("Finished building the interpolators.", flush=True)

    def __call__(self, r, theta, phi):
        """
        Evaluate the covariance matrix for the given galaxy positions.

        Parameters
        ----------
        r, theta, phi : 1-dimensional arrays of floats
            The galaxy positions in spherical coordinates (r, theta, phi).

        Returns
        -------
        2-dimensional array of floats
        """
        Main.r = np.asarray(r, dtype=float)
        Main.theta = np.asarray(theta, dtype=float)
        Main.phi = np.asarray(phi, dtype=float)
        return Main.eval(
            "pecvel_covmat_from_interp(r, theta, phi, Cij_interp, Cii_interp)")  # noqa