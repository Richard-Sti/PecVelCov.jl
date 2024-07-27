#=
Script to calculate the covariance matrix of the peculiar velocity field for Quijote haloes.
=#
using JLD2
using PecVelCov


function cartesian_to_spherical(x::Real, y::Real, z::Real)
    r = (x^2 + y^2 + z^2)^0.5
    θ = acos(z / r)
    ϕ = atan(y, x)

    if ϕ < 0
        ϕ += 2 * π
    end

    return r, θ, ϕ
end


function cartesian_to_spherical(x, y, z)
    results = cartesian_to_spherical.(x, y, z)
    r = [res[1] for res in results]
    θ = [res[2] for res in results]
    ϕ = [res[3] for res in results]

    return r, θ, ϕ
end

function main()
    nsims = [0]
    nobs = 27
    ell_min = 0

    fname_input(nsim) = "/mnt/extraspace/rstiskalek/quijote/BulkFlow_fiducial/BF_nsim_$nsim.hdf5"
    fname_output(nsim) = "/mnt/extraspace/rstiskalek/quijote/BulkFlow_fiducial/BF_nsim_covmat_ell_min_$(ell_min)_$(nsim).hdf5"

    println("Building interpolators...")
    Cii_interp = build_Cii_interpolator("/mnt/extraspace/rstiskalek/BBF/Cii_grid_ellmin_$ell_min.jld2");
    Cij_interp = build_Cij_joint_interpolator(
        "/mnt/extraspace/rstiskalek/BBF/Cij_grid_ellmin_$ell_min.jld2",
        "/mnt/extraspace/rstiskalek/BBF/Cij_close_grid_ellmin_$ell_min.jld2",
        "/mnt/extraspace/rstiskalek/BBF/Cij_opposite_grid_ellmin_$ell_min.jld2")


    for i in nsims
        # Remove output file if it already exists
        if isfile(fname_output(i))
            rm(fname_output(i))
        end

        for j in 0:nobs-1
            println("Processing simulation $i, observer $j...")
            # Read in halo positions and convert to spherical coordinates
            r, θ, ϕ = nothing, nothing, nothing
            jldopen(fname_input(i)) do file
                pos = file["obs_$j"]["halo_pos"]
                # obs = file["obs_$j"]["observer"]
                # x, y, z = pos[1, :] .- obs[1], pos[2, :] .- obs[2], pos[3, :] .- obs[3]
                x, y, z = pos[1, :], pos[2, :], pos[3, :]
                r, θ, ϕ = cartesian_to_spherical(x, y, z)
            end

            Σ = pecvel_covmat_from_interp(r, θ, ϕ, Cij_interp, Cii_interp)

            jldopen(fname_output(i), "a+") do file
                file["obs_$j"] = Σ
            end

        end
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
