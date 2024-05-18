#=
Script to sample the covariance matrix C_ij over a grid of r, r', and cosθ values. The underlying cosmological
choice is banked in through the power spectrum interpolator. The script saves the computed covariance matrix elements.
=#
using ArgParse
using Base.Threads
using JLD2
using PecVelCov
using ProgressMeter


###############################################################################
#                          Parse the command line                             #
###############################################################################


function get_cmd(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--runtype"
            help = "Type of run: `full`, `diagonal`, `close`, or `opposite"
            arg_type = String
        "--continue"
            help = "Whether to continue the computation from the last saved point."
            arg_type = Bool
            default = false
        "--include_dipole"
            help = "Include the dipole term in the covariance matrix"
            arg_type = Bool
            default = false
    end


    args = parse_args(args, s)
    args["fname_pk"] = "/mnt/users/rstiskalek/BayesianBulkFlows/data/pk_fiducial.npy"
    args["ks"] = make_spacing(1e-4, 10, 1024, 0.33; log_fraction=0.33)
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
        println("Sampling Σ_ij over $(length(args["rs"])) x $(length(args["rs"])) x $(length(args["cosθs"])) grid points.")
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

    args["fname_temp"] = replace(args["fname_out"], ".jld2" => "_temp.jld2")

    flush(stdout)
    return args
end


###############################################################################
#                          Main computation functions                         #
###############################################################################


function run_full(rs, cosθs, Pk, ks, ell_min, ell_max, djn_interp, start_krs, fname_temp)
    nr, ncosθ = length(rs), length(cosθs)

    kstart = 1
    jldopen(fname_temp, "r") do file
        kstart = file["kstart"]
        if kstart > 1
            println("Resuming computation from k = $(kstart)."); flush(stdout)
        end
    end

    p = Progress(nr * nr * ncosθ, dt=1)
    for k in kstart:ncosθ
        cosθ = cosθs[k]
        Pells = precompute_legendre_Pells(ell_min, ell_max, cosθ)
        Cij_current = fill(NaN, nr, nr)
        # Parallelize the inner loop for thread safety when writing the temporary file.
        @threads for i in 1:nr
            kri = ks .* rs[i]
            for j in 1:nr
                krj = ks .* rs[j]
                Cij_current[i, j] = C_ij(kri, krj, Pells, Pk, ks; ell_max=ell_max, djn_interp=djn_interp, start_krs=start_krs)
                next!(p)
            end
            flush(stdout)
        end

        # Save the computed elements to a temporary file in case the computation is interrupted.
        jldopen(fname_temp, "r+") do file
            file["Σ_$k"] = Cij_current
            delete!(file, "kstart")
            file["kstart"] = k + 1
        end

    end
    finish!(p)

    # Now concatenate the computed elements from the temporary files into a single array.
    Cij_grid = zeros(nr, nr, ncosθ)
    jldopen(fname_temp, "r") do file
        for k in 1:ncosθ
            Cij_grid[:, :, k] = file["Σ_$k"]
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


###############################################################################
#                                 Main script                                #
###############################################################################


if abspath(PROGRAM_FILE) == @__FILE__
    println("We are running with $(Threads.nthreads()) threads.\n")
    args = get_cmd(ARGS)

    # Diagonal is fast so we don't need to ever continue it.
    if args["continue"] && args["runtype"] == "diagonal"
        error("Cannot continue the computation for the diagonal elements.")
    end

    if args["continue"]
        if !isfile(args["fname_temp"])
            error("File `$(args["fname_out"])` does not exist.")
        end

    else
        if isfile(args["fname_temp"])
            println("File `$(args["fname_temp"])` already exists. Removing it.")
            rm(args["fname_temp"])
        end

        # Write a new temporary file
        jldopen(args["fname_temp"], "w") do file
            file["rs"] = args["rs"]
            file["cosθs"] = args["cosθs"]
            file["kstart"] = 1
        end
    end

    println("Loading power spectrum from `$(args["fname_pk"])`..."), flush(stdout)
    ks = args["ks"]
    Pk = build_Pk_interpolator(args["fname_pk"]).(ks)

    println("Loading the precomputed spherical Bessel function derivatives from `$(args["fname_djn"])`..."), flush(stdout)
    djn_interp, start_krs = build_dnj_interpolator(args["fname_djn"])


    println("Starting computation, kind is `$(args["runtype"])`."), flush(stdout)
    if args["runtype"] == "diagonal"
        Σ = run_diagonal(args["rs"], Pk, ks, args["ell_min"], args["ell_max"], djn_interp, start_krs)
    else
        Σ = run_full(args["rs"], args["cosθs"], Pk, ks, args["ell_min"], args["ell_max"], djn_interp, start_krs, args["fname_temp"])
    end

    println("Saving computed covariance matrix elements to `$(args["fname_out"])`.")
    jldopen(args["fname_out"], "w") do file
        file["rs"] = args["rs"]
        file["cosθs"] = args["cosθs"]
        file["Σ"] = Σ
    end

    if isfile(args["fname_temp"])
        println("Removing temporary file `$(args["fname_temp"])`.")
        rm(args["fname_temp"])
    end
end
