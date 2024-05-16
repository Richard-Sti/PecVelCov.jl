#=
Script to sample the covariance matrix C_ij over a grid of r, r', and cosθ values. The underlying cosmological
choice is banked in through the power spectrum interpolator. The script saves the computed covariance matrix elements.
=#
using ArgParse
using Base.Threads
using JLD2
using PecVelCov
using ProgressMeter


function get_cmd(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--runtype"
            help = "Type of run: `full`, `diagonal`, `close`, or `opposite"
            arg_type = String
        "--include_dipole"
            help = "Include the dipole term in the covariance matrix"
            arg_type = Bool
            default = false
        "--logk_range"
            help = "Range of log(k) values [h / Mpc]"
            arg_type = Float64
            nargs = 2
            default = [-4., 1.]
        "--npoints_k"
            help = "Number of points in k"
            arg_type = Int
            default = 4096
    end

    args = parse_args(args, s)

    args["fname_pk"] = "/mnt/users/rstiskalek/BayesianBulkFlows/data/pk_fiducial.npy"
    args["fname_djn"] = "/mnt/extraspace/rstiskalek/BBF/djn_grid.jld2"
    args["ell_min"] = args["include_dipole"] ? 1 : 2

    if args["runtype"] == "full"
        args["fname_out"] = "/mnt/extraspace/rstiskalek/BBF/Cij_grid.jld2"
        args["rs"] = LinRange(0.1, 350, 500)  # Mpc / h
        args["cosθs"] = LinRange(-0.93, 0.93, 500)
        args["ell_max"] = 20
        println("Sampling Σ_ij over $(length(args["rs"])) x $(length(args["rs"])) x $(length(args["cosθs"])) grid points.")
    elseif args["runtype"] == "diagonal"
        args["fname_out"] = "/mnt/extraspace/rstiskalek/BBF/Cii_grid.jld2"
        args["rs"] = LinRange(0.1, 350, 50000)  # Mpc / h
        args["cosθs"] = nothing
        args["ell_max"] = 100
        println("Sampling Σ_ii over $(length(args["rs"])) grid points.")
    elseif args["runtype"] == "close"
        args["fname_out"] = "/mnt/extraspace/rstiskalek/BBF/Cij_close_grid.jld2"
        args["rs"] = LinRange(0.1, 350, 500)  # Mpc / h
        args["cosθs"] = LinRange(0.925, 1, 500)
        args["ell_max"] = 100
        println("Sampling Σ_ij over $(length(args["rs"])) x $(length(args["rs"])) x $(length(args["cosθs"])) grid points.")
    elseif args["runtype"] == "opposite"
        args["fname_out"] = "/mnt/extraspace/rstiskalek/BBF/Cij_opposite_grid.jld2"
        args["rs"] = LinRange(0.1, 350, 500)  # Mpc / h
        args["cosθs"] = LinRange(-1, -0.925, 500)
        args["ell_max"] = 100
        println("Sampling Σ_ij over $(length(args["rs"])) x $(length(rs)) x $(length(args["cosθs"])) grid points.")
    else
        error("Invalid runtype: $(args["runtype"])")
    end

    println("Spacing in r: $(args["rs"][2] - args["rs"][1]) [Mpc / h].")
    if args["cosθs"] !== nothing
        println("Spacing in cosθ: $(args["cosθs"][2] - args["cosθs"][1]).")
    end

    if args["include_dipole"]
        args["fname_out"] = replace(args["fname_out"], ".jld2" => "_including_dipole.jld2")
    end

    flush(stdout)
    return args
end


function run_full(rs, cosθs, Pk, ks, ell_min, ell_max, djn_interp, start_krs)
    nr, ncosθ = length(rs), length(cosθs)
    Cij_grid = zeros(nr, nr, ncosθ)

    @showprogress dt=1 @threads for k in reverse(1:ncosθ)
        cosθ = cosθs[k]
        Pells = precompute_legendre_Pells(ell_min, ell_max, cosθ)
        for i in 1:nr
            kri = ks .* rs[i]
            for j in 1:nr
                krj = ks .* rs[j]
                Cij_grid[i, j, k] = C_ij(kri, krj, Pells, Pk, ks; ell_max=ell_max, djn_interp=djn_interp, start_krs=start_krs)
            end
        end
    end

    return Cij_grid
end


function run_diagonal(rs, Pk, ks, ell_min, ell_max, djn_interp, start_krs)
    nr = length(rs)
    Cii_grid = zeros(nr)

    Pells = Dict(ell => 1. for ell in ell_min:ell_max)

    @showprogress dt=1 @threads for i in 1:nr
        kri = ks .* rs[i]
        Cii_grid[i] = C_ij(kri, kri, Pells, Pk, ks; ell_max=ell_max, djn_interp=djn_interp, start_krs=start_krs)
    end

    return Cii_grid
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("We are running with $(Threads.nthreads()) threads.\n")
    args = get_cmd(ARGS)

    println("Loading power spectrum from `$(args["fname_pk"])`...")
    ks = 10 .^ LinRange(args["logk_range"][1], args["logk_range"][2], args["npoints_k"])
    Pk = build_Pk_interpolator(args["fname_pk"]).(ks)

    println("Loading the precomputed spherical Bessel function derivatives from `$(args["fname_djn"])`...")
    djn_interp, start_krs = build_dnj_interpolator(args["fname_djn"])


    println("Starting computation of covariance matrix elements...\n")
    if args["cosθs"] === nothing
        Σ = run_diagonal(args["rs"], Pk, ks, args["ell_min"], args["ell_max"], djn_interp, start_krs)
    else
        Σ = run_full(args["rs"], args["cosθs"], Pk, ks, args["ell_min"], args["ell_max"], djn_interp, start_krs)
    end

    println("Saving computed covariance matrix elements to `$(args["fname_out"])`.")
    jldsave(args["fname_out"], rs=args["rs"], cosθs=args["cosθs"], Σ=Σ)
end
