module PecVelCov

import GSL: sf_legendre_Pl
import Interpolations: LinearInterpolation
import NPZ: npzread
import Integrals: SimpsonsRule, SampledIntegralProblem, solve
import SpecialFunctions: sphericalbesselj
import JLD2: jldopen

export build_Pk_interpolator, djn, sf_legendre_Pl, C_ij, djn2


###############################################################################
#                     Building interpolators                                  #
###############################################################################


"""
    build_Pk_interpolator(fname::String) -> LinearInterpolation

Build a linear interpolator for the power spectrum `P(k)` from the data stored in the file `fname`.
"""
function build_Pk_interpolator(fname)
    X = npzread(fname)
    k = X[:, 1]
    Pk = X[:, 2]

    return LinearInterpolation(k, Pk)
end


###############################################################################
#                   Covariance matrix elements calculation                    #
###############################################################################


"""
    djn(n::Integer, x::Number) -> Number

Compute the derivative of the spherical Bessel function of the first kind `jₙ(x)` with respect to `x`.
"""
djn(n, x) = n / x * sphericalbesselj(n, x) - sphericalbesselj(n + 1, x)


"""
    C_ij(ri, rj, cosθ, Pk, ks; ell_min=0, ell_max=20) -> Number

Calculate the covariance matrix element `C_{ij}` for the two points `rᵢ` and `rⱼ` separated by
an angle `θ`.

# TODO: Still missing prefactors before the integral over k.
"""
function C_ij(ri, rj, cosθ, Pk, ks; ell_min=0, ell_max=20)
    ys_dummy = zeros(length(ks))
    for i in eachindex(ks)
        kri = ks[i] * ri
        krj = ks[i] * rj
        for ell in ell_min:ell_max
            ys_dummy[i] += (ell + 1) * djn(ell, kri) * djn(ell, krj) * sf_legendre_Pl(ell, cosθ)
        end

        ys_dummy[i] *= Pk[i]
    end

    sol = solve(SampledIntegralProblem(ys_dummy, ks), SimpsonsRule())
    return sol.u
end



end
