#=
Script to precompute the spherical Bessel functions djn(ell, x) over a grid of ell and x values.
=#
using ArgParse
using JLD2
using ProgressMeter
using PecVelCov

function get_cmd(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--ell_min"
            help = "Minimum ell"
            arg_type = Int
            default = 1
        "ell_max"
            help = "Maximum ell"
            arg_type = Int
            default = 100
        "logk_range"
            help = "Range of log(k) values [h / Mpc]"
            arg_type = Float64
            nargs = 2
            default = [-4, 1]
        "r_range"
            help = "Range of r values [Mpc / h]"
            arg_type = Float64
            nargs = 2
            default = [0.1, 350]
        "npoints"
            help = "Number of points in kr"
            arg_type = Int
            default = 524288
    end

    return parse_args(args, s)
end


args = get_cmd(ARGS)

xmin, xmax = [10^args["logk_range"][i] * args["r_range"][i] for i in 1:2]
xs = 10 .^ LinRange(log10(xmin), log10(xmax), args["npoints"])
ells = args["ell_min"]:args["ell_max"]
ys = zeros(length(xs), length(ells))

@showprogress dt=0.1 for (i, ell) in enumerate(ells)
    ys[:, i] = djn.(ell, xs)
end

fname_out = "/mnt/extraspace/rstiskalek/BBF/djn_grid.jld2"
println("Saving to `$fname_out`.")
jldsave(fname_out, xs=xs, ells=ells, ys=ys)
