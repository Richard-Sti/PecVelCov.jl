using PecVelCov
using ProgressMeter
using Base.Threads
using JLD2

include_dipole = false
ell_min = include_dipole ? 1 : 2
# Define the grid over which to sample the covariance matrix, and the k values
rs = LinRange(0.1, 350, 100)  # Mpc / h
cosθs = LinRange(-1, 1, 100)
ks = 10 .^ LinRange(-4, 1, 2048)
println("Sampling covariance matrix over $(length(rs)) x $(length(rs)) x $(length(cosθs)) grid points.")
println("Spacing in r: $(rs[2] - rs[1]) Mpc / h")
println("Spacing in cosθ: $(cosθs[2] - cosθs[1])")

# Load the power spectrum interpolator
fname_pk = "/mnt/users/rstiskalek/BayesianBulkFlows/data/pk_fiducial.npy"
println("Loading power spectrum from `$fname_pk`...")
pk_interp = build_Pk_interpolator(fname_pk)
pk = pk_interp(ks)
println("Power spectrum loaded. Ranges from $(minimum(ks)) to $(maximum(ks)).")


nrs = length(rs)
ncosθ = length(cosθs)
Cij_grid = zeros(nrs, nrs, ncosθ)

println("We are running with $(Threads.nthreads()) threads.")
@showprogress dt=1 desc="Computing C_ij" @threads for idx in 1:(nrs^2 * ncosθ)
    # Determine the original indices on the 3D grid
    i = ((idx - 1) ÷ (nrs * ncosθ)) + 1
    j = (((idx - 1) ÷ ncosθ) % nrs) + 1
    k = ((idx - 1) % ncosθ) + 1

    cosθ = cosθs[k]
    ell_max = cosθ < 0.95 ? 50 : 100
    Cij_grid[i, j, k] = C_ij(rs[i], rs[j], cosθ, pk, ks; ell_min=ell_min, ell_max=ell_max)
end


fname_out = "/mnt/extraspace/rstiskalek/BBF/Cij_grid.jld2"
if include_dipole
    fname_out = replace(fname_out, ".jld2" => "_including_dipole.jld2")
end
println("Saving computed covariance matrix elements to `$fname_out`.")
jldsave(fname_out, rs=rs, cosθs=cosθs, Cij_grid=Cij_grid)
