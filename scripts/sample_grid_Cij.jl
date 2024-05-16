using PecVelCov
using ProgressMeter
using Base.Threads
using JLD2

include_dipole = false
ell_min = include_dipole ? 1 : 2
global_ell_max = 100

# Define the grid over which to sample the covariance matrix, and the k values
rs = LinRange(0.1, 350, 400)  # Mpc / h
cosθs = LinRange(-1, 1, 400)
ks = 10 .^ LinRange(-4, 1, 2 * 2048)
println("Sampling covariance matrix over $(length(rs)) x $(length(rs)) x $(length(cosθs)) grid points.")
println("Spacing in r: $(rs[2] - rs[1]) Mpc / h")
println("Spacing in cosθ: $(cosθs[2] - cosθs[1])")

# Load the power spectrum interpolator
fname_pk = "/mnt/users/rstiskalek/BayesianBulkFlows/data/pk_fiducial.npy"
println("Loading power spectrum from `$fname_pk`...")
pk_interp = build_Pk_interpolator(fname_pk)
pk = pk_interp(ks)
println("Power spectrum loaded. Ranges from $(minimum(ks)) to $(maximum(ks)).")


# Precompute the derivatives of the spherical Bessel functions
println("Precomuting the derivatives of the spherical Bessel functions...")
kmin, kmax = minimum(ks), maximum(ks)
rmin, rmax = minimum(rs), maximum(rs)
krs = 10 .^ LinRange(log10(kmin * rmin), log10(kmax * rmax), 262144)
djn_interp = precompute_djn(ell_min, global_ell_max, krs);
start_krs = precompute_djn_start(djn_interp, krs)

nr, ncosθ = length(rs), length(cosθs)
Cij_grid = zeros(nr, nr, ncosθ)

println("We are running with $(Threads.nthreads()) threads.\n")
@showprogress dt=1 @threads for k in reverse(1:ncosθ)
    cosθ = cosθs[k]
    ell_max = cosθ < 0.95 ? 20 : 100
    Pells = precompute_legendre_Pells(ell_min, ell_max, cosθ)
    for i in 1:nr
        kri = ks .* rs[i]
        for j in 1:nr
            krj = ks .* rs[j]
            Cij_grid[i, j, k] = C_ij(kri, krj, Pells, pk, ks; ell_max=ell_max, djn_interp=djn_interp, start_krs=start_krs)
        end
    end
end


fname_out = "/mnt/extraspace/rstiskalek/BBF/Cij_grid.jld2"
if include_dipole
    fname_out = replace(fname_out, ".jld2" => "_including_dipole.jld2")
end
println("Saving computed covariance matrix elements to `$fname_out`.")
jldsave(fname_out, rs=rs, cosθs=cosθs, Cij_grid=Cij_grid)
