module PecVelCov

import GSL: sf_legendre_Pl
import Integrals: SampledIntegralProblem, SimpsonsRule, solve
import JLD2: jldopen
import NPZ: npzread
import SpecialFunctions: sphericalbesselj
using ProgressMeter
using LinearAlgebra
using Interpolations

export build_Cij_interpolator, build_Cij_joint_interpolator, build_Cii_interpolator, build_dnj_interpolator,
    build_Pk_interpolator, covariance_inverse_and_determinant, C_ij, djn, make_spacing, sf_legendre_Pl,
    precompute_legendre_Pells, pecvel_covmat_from_interp, pecvel_covmat_brute


###############################################################################
#                     Building interpolators                                  #
###############################################################################


"""
    make_spacing(xmin, xmax, npoints, transition; log_fraction=0.5) -> Vector

Create a spacing of `npoints` points between `xmin` and `xmax` with a transition point at `transition`.
The spacing is logarithmic between `xmin` and `transition` and linear between `transition` and `xmax`.
"""
function make_spacing(xmin, xmax, npoints, transition; log_fraction=0.5)
    if log_fraction > 1 || log_fraction < 0
        throw(ArgumentError("log_fraction must be between 0 and 1"))
    end

    if transition < xmin || transition > xmax
        throw(ArgumentError("transition must be between xmin and xmax"))
    end

    npoints_log = Int(floor(npoints * log_fraction))
    xlow = 10 .^ LinRange(log10(xmin), log10(transition), npoints_log + 1)
    xhigh = LinRange(transition, xmax, npoints - npoints_log)
    return vcat(xlow[1:end - 1], xhigh)

end


"""
    build_Pk_interpolator(fname::String) -> LinearInterpolation

Build a linear interpolator for the power spectrum `P(k)` from the data stored in the file `fname`.
Internally, the interpolator is built in log-log space.
"""
function build_Pk_interpolator(fname)
    X = npzread(fname)
    k = X[:, 1]
    Pk = X[:, 2]

    finterp = LinearInterpolation(log.(k), log.(Pk))
    return x -> exp(finterp(log(x)))
end


"""
    build_Cij_interpolator(fname::String) -> Interpolations.ScaledInterpolation

Build an interpolator for the covariance matrix elements `C_{ij}` from the data stored in the file `fname`.
"""
function build_Cij_interpolator(fname)
    rs, cosθs, Cij_grid = nothing, nothing, nothing
    jldopen(fname, "r") do file
        rs = file["rs"]
        cosθs = file["cosθs"]
        Cij_grid = file["Σ"]
    end

    interp = interpolate(Cij_grid, BSpline(Cubic(Line(OnGrid()))))
    interp = scale(interp, rs, rs, cosθs)
    return extrapolate(interp, Line())
end


"""
    build_Cij_joint_interpolator(fname_full::String, fname_close::String, fname_opposite::String) -> Function

Build a joint interpolator for the covariance matrix elements `C_{ij}` from the data stored in the files `fname_full`,
`fname_close`, and `fname_opposite`. Choice of the appropriate interpolator is based on the hard-coded value of `cosθ`.
"""
function build_Cij_joint_interpolator(fname_full, fname_close, fname_opposite)
    Cij_interp_full = build_Cij_interpolator(fname_full)
    Cij_interp_close = build_Cij_interpolator(fname_close)
    Cij_interp_opposite = build_Cij_interpolator(fname_opposite)

    function Cij_joint_interp(r1, r2, cosθ)
        if cosθ > 0.925
            return Cij_interp_close(r1, r2, cosθ)
        elseif cosθ < -0.925
            return Cij_interp_opposite(r1, r2, cosθ)
        else
            return Cij_interp_full(r1, r2, cosθ)
        end
    end

    return Cij_joint_interp

end


"""
    build_Cii_interpolator(fname::String) -> Interpolations.ScaledInterpolation

Build an interpolator for the covariance matrix diagonal elements `C_{ii}` from the data stored in the file `fname`.
"""
function build_Cii_interpolator(fname)
    rs, Cii_grid = nothing, nothing, nothing
    jldopen(fname, "r") do file
        rs = file["rs"]
        Cii_grid = file["Σ"]
    end

    interp = interpolate(Cii_grid, BSpline(Cubic(Line(OnGrid()))))
    interp = scale(interp, rs)
    return extrapolate(interp, Line())
end


"""
    precompute_legendre_Pells(ell_min, ell_max, cosθ) -> Dict

Precompute the Legendre polynomials for the range of multipoles `ell_min` to `ell_max` at `cosθ`.
"""
function precompute_legendre_Pells(ell_min, ell_max, cosθ)
    return Dict(ell => sf_legendre_Pl(ell, cosθ) for ell in ell_min:ell_max)
end


"""
    build_dnj_interpolator(fname::String) -> Tuple{Dict, Dict}

Build a precomputed interpolator for the derivative of the spherical Bessel function of the first kind
`jₙ(x)` with respect to `x` for the range of multipoles `ell_min` to `ell_max`.
"""
function build_dnj_interpolator(fname)
    xs, ells, ys = nothing, nothing, nothing
    jldopen(fname, "r") do file
        xs = file["xs"]
        ells = file["ells"]
        ys = file["ys"]
    end

    dnj_interp = Dict(ell => interpolate((xs,), ys[:, i], Gridded(Linear())) for (i, ell) in enumerate(ells))
    start_xs = Dict(ell => xs[findfirst(x -> abs(x) > 1e-10, ys[:, i])] for (i, ell) in enumerate(ells))
    return dnj_interp, start_xs
end


###############################################################################
#                   Covariance matrix elements calculation                    #
###############################################################################


"""
    djn(n::Integer, x::Number) -> Number

Derivative of the spherical Bessel function of the first kind `jₙ(x)` with respect to `x`.
"""
djn(n, x) = n / x * sphericalbesselj(n, x) - sphericalbesselj(n + 1, x)


"""
    C_ij(kri, krj, Pells, Pk, ks; ell_min=2, ell_max=20, djn_interp=nothing, start_krs=nothing) -> Number

Covariance matrix element `C_{ij}` for the two points `rᵢ` and `rⱼ` separated by
an angle `θ`.

# TODO: Still missing prefactors before the integral over k.
"""
function C_ij(kri, krj, Pells, Pk, ks; ell_min=2, ell_max=20, djn_interp=nothing, start_krs=nothing)
    ys_dummy = zeros(length(ks))
    ell_range = ell_min:ell_max
    djn_eval = isa(djn_interp, Nothing) ? (ell, k) -> djn(ell, k) : (ell, k) -> djn_interp[ell](k)
    start_krs_eval = isa(start_krs, Nothing) ? ell -> 0.0 : ell -> start_krs[ell]

    @inbounds for i in eachindex(ks)
        y_val, kri_i, krj_i = 0.0, kri[i], krj[i]
        for ell in ell_range
            # Once this condition is trigerred it will always be trigerred for
            # the rest of the loop over ell.
            start_kr = start_krs_eval(ell)
            if kri_i < start_kr || krj_i < start_kr
                break
            end
            y_val += @fastmath (2 * ell + 1) * djn_eval(ell, kri_i) * djn_eval(ell, krj_i) * Pells[ell]
        end
        ys_dummy[i] = y_val * Pk[i]
    end

    return solve(SampledIntegralProblem(ys_dummy, ks), SimpsonsRule()).u
end


###############################################################################
#                   Covariance matrix for a set of tracers                    #
###############################################################################


"""
    pecvel_covmat_from_interp(rs, θs, ϕs, Pk, ks, Cij_interpolator, Cii_interpolator; ell_min, djn_interp, start_krs) -> Matrix

Compute the covariance matrix for the peculiar velocity field for a set of tracers using the
interpolator `Cij_interpolator` for the off-diagonal covariance matrix elements and `Cii_interpolator`
for the diagonal elements.
"""
function pecvel_covmat_from_interp(rs, θs, ϕs, Pk, ks, Cij_interpolator, Cii_interpolator; ell_min, djn_interp, start_krs)
    @assert length(rs) == length(θs) && length(θs) == length(ϕs) "rs, θs, and ϕs must have the same length."
    sinθs, cosθs = sin.(θs), cos.(θs)

    Σ = zeros(length(rs), length(rs))
    # First compute the off-diagonal elements.
    @showprogress dt = 0.1 for i in eachindex(rs)
        sinθ_i, cosθ_i, ϕ_i = sinθs[i], cosθs[i], ϕs[i]
        for j in eachindex(rs)
            if j > i
                continue
            end

            cosΔ = min(sinθ_i * sinθs[j] * cos(ϕ_i - ϕs[j]) + cosθ_i * cosθs[j], 1.0)
            Σij = Cij_interpolator(rs[i], rs[j], cosΔ)

            Σ[i, j] = Σij
            Σ[j, i] = Σij
        end
    end

    # And then compute the diagonal elements.
    for i in eachindex(rs)
        Σ[i, i] = Cii_interpolator(rs[i])
    end

    return Σ
end


"""
    pecvel_covariance_matrix_brute(rs, θs, ϕs, Pk, ks; ell_min=2, djn_interp=nothing, start_krs=nothing) -> Matrix

Compute the covariance matrix for the peculiar velocity field for a set of tracers using a brute-force, without
any interpolation of C_ij elements.
"""
function pecvel_covmat_brute(rs, θs, ϕs, Pk, ks; ell_min=2, djn_interp=nothing, start_krs=nothing)
    @assert length(rs) == length(θs) && length(θs) == length(ϕs) "rs, θs, and ϕs must have the same length."
    sinθs, cosθs = sin.(θs), cos.(θs)

    Σ = zeros(length(rs), length(rs))
    @showprogress dt = 0.1 for i in eachindex(rs)
        kri = ks .* rs[i]
        sinθ_i, cosθ_i = sinθs[i], cosθs[i]
        ϕ_i = ϕs[i]

        for j in eachindex(rs)
            if j > i
                continue
            end

            krj = ks .* rs[j]
            cosΔ = min(sinθ_i * sinθs[j] * cos(ϕ_i - ϕs[j]) + cosθ_i * cosθs[j], 1.0)

            ell_max = cosΔ < 0.95 ? 20 : 100
            Pells = precompute_legendre_Pells(ell_min, ell_max, cosΔ)

            Σij = C_ij(kri, krj, Pells, Pk, ks; ell_max=ell_max, djn_interp=djn_interp, start_krs=start_krs)
            Σ[i, j] = Σij
            Σ[j, i] = Σij
        end
    end
    return Σ

end


###############################################################################
#                 Covariance matrix inverse and determinant                   #
###############################################################################


"""
    covariance_inverse_and_determinant(C) -> Tuple{Matrix, Number}

Compute the inverse and determinant of the covariance matrix `C` using the Cholesky decomposition.
"""
function covariance_inverse_and_determinant(C)
    L = cholesky(C).L
    L_inv = inv(L)

    det = prod(diag(L))^2

    C_inv = L_inv' * L_inv

    return C_inv, det
end

end
