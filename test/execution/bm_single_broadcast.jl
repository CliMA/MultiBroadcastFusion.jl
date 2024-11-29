#=
using Revise; include(joinpath("test", "execution", "bm_single_broadcast.jl"))
=#

include("utils_test.jl")
include("utils_setup.jl")
include("utils_benchmark.jl")

import MultiBroadcastFusion as MBF

function perf_kernel_single_kernel_unfused!(X, Y)
    (; x1, x2, x3, x4) = X
    (; y1, y2, y3, y4) = Y
    # Total: 1 writes, 4 reads
    @. y1 = x1 + x3 + x2 + x4
end

function perf_kernel_single_kernel_fused!(X, Y)
    (; x1, x2, x3, x4) = X
    (; y1, y2, y3, y4) = Y
    # Total: 1 writes, 4 reads
    MBF.@fused_direct begin
        @. y1 = x1 + x3 + x2 + x4
    end
end

@static get(ENV, "USE_CUDA", nothing) == "true" && using CUDA
use_cuda = @isdefined(CUDA) && CUDA.has_cuda() # will be true if you first run `using CUDA`
AType = use_cuda ? CUDA.CuArray : Array
device_name = use_cuda ? CUDA.name(CUDA.device()) : "CPU"
bm = Benchmark(; device_name, float_type = Float32)
problem_size = (50, 5, 5, 6, 5400)

array_size = problem_size # array
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(
    use_cuda;
    unfused! = perf_kernel_single_kernel_unfused!,
    fused! = perf_kernel_single_kernel_fused!,
    X,
    Y,
)
# Benchmark
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_single_kernel_unfused!,
    X,
    Y;
    n_reads_writes = 1 + 4,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_single_kernel_fused!,
    X,
    Y;
    n_reads_writes = 1 + 4,
    problem_size = array_size,
)

array_size = (prod(problem_size),) # vector
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(
    use_cuda;
    unfused! = perf_kernel_single_kernel_unfused!,
    fused! = perf_kernel_single_kernel_fused!,
    X,
    Y,
)
# Benchmark
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_single_kernel_unfused!,
    X,
    Y;
    n_reads_writes = 1 + 4,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_single_kernel_fused!,
    X,
    Y;
    n_reads_writes = 1 + 4,
    problem_size = array_size,
)


tabulate_benchmark(bm)

nothing