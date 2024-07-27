queue="redwood"
env="/usr/local/shared/julia/julia-1.10.0/bin/julia"
fname="sample_grid_Cij.jl"

nthreads=64
memory=7

# Runtype: diagonal, full, opposite, close
runtype=${1}
ell_min=${2}
continue=${3:-false}

# Check that runtype is OK
if [ "$runtype" != "diagonal" ] && [ "$runtype" != "full" ] && [ "$runtype" != "opposite" ] && [ "$runtype" != "close" ] && [[ "$runtype" != *"fixed_radius"* ]]; then
    echo "Invalid runtype (1): $runtype"
    exit 1
fi

# Diagonal is not parallelizable
if [ "$runtype" == "diagonal" ] || [[ "$runtype" == *"fixed_radius"* ]]; then
    nthreads=1
fi

# Check if ell_min is valid
if [ "$ell_min" -ne 0 ] && [ "$ell_min" -ne 1 ] && [ "$ell_min" -ne 2 ] && [ "$ell_min" -ne 3 ]; then
    echo "Invalid ell_min (2): $ell_min"
    exit 1
fi

if [ "$continue" != "true" ] && [ "$continue" != "false" ]; then
    echo "Error: continue (3) must be 'true' or 'false'. By default, it is 'false'."
    exit 1
fi


if [ "$continue" = true ]; then
    cmd="$env --threads $nthreads $fname --runtype $runtype --ell_min $ell_min --continue true"
else
    cmd="$env --threads $nthreads $fname --runtype $runtype --ell_min $ell_min"
fi


cmd_submit="addqueue -s -q $queue -n 1x$nthreads -m $memory $cmd"
echo "Submitting job with command:"
echo $cmd_submit

eval $cmd_submit
