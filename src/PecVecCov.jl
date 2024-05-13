module PecVecCov

import GSL: sf_bessel_jl, sf_legendre_Pl
import Interpolations: LinearInterpolation
import NPZ: npzread
import Integrals: SimpsonsRule, SampledIntegralProblem, solve
import SpecialFunctions: sphericalbesselj

export build_Pk_interpolator, djn, sf_legendre_Pl, C_ij!, djn2


"""
    djn(n::Integer, x::Number) -> Number

Compute the derivative of the spherical Bessel function of the first kind `jₙ(x)` with respect to `x`.
"""
djn(n, x) = n / x * sf_bessel_jl(n, x) - sf_bessel_jl(n + 1, x)


"""
    djn2(n::Integer, x::Number) -> Number

More numerically stable version of `djn` but slower.
"""
djn_stable(n, x) = n / x * sphericalbesselj(n, x) - sphericalbesselj(n + 1, x)


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


"""
    C_ij!(ys_dummy, ri, rj, cosθ, Pk, ks; ell_min=0, ell_max=20) -> Number

Calculate the covariance matrix element `C_{ij}` for the two points `rᵢ` and `rⱼ` separated by
an angle `θ`. `ys_dummy` is a dummy array that will be used to store the integrand for
the integral over the wavenumber `k`.


# TODO: Still missing prefactors before the integral over k.
"""
function C_ij!(ys_dummy, ri, rj, cosθ, Pk, ks; ell_min=0, ell_max=20)
    for i in eachindex(ks)
        ys_dummy[i] = 0.
        kri = ks[i] * ri
        krj = ks[i] * rj
        for ell in ell_min:ell_max
            if ell > 50
                ys_dummy[i] += (ell + 1) * djn_stable(ell, kri) * djn_stable(ell, krj) * sf_legendre_Pl(ell, cosθ)
            else
                ys_dummy[i] += (ell + 1) * djn(ell, kri) * djn(ell, krj) * sf_legendre_Pl(ell, cosθ)
            end
        end

        ys_dummy[i] *= Pk[i]
    end

    sol = solve(SampledIntegralProblem(ys_dummy, ks), SimpsonsRule())
    return sol.u
end

end
