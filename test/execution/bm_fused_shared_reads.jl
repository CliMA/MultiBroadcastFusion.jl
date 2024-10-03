#=
using Revise; include(joinpath("test", "execution", "bm_fused_shared_reads.jl"))
=#

include("utils_test.jl")
include("utils_setup.jl")
include("utils_benchmark.jl")

import MultiBroadcastFusion as MBF

function perf_kernel_shared_reads_unfused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    @. y1 = x1 + x2 + x3 + x4
    @. y2 = x2 + x3 + x4 + x5
    @. y3 = x3 + x4 + x5 + x6
    @. y4 = x4 + x5 + x6 + x7
    @. y5 = x5 + x6 + x7 + x8
    @. y6 = x6 + x7 + x8 + x9
    @. y7 = x7 + x8 + x9 + x10
end

function perf_kernel_shared_reads_fused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    MBF.@fused_direct begin
        @. y1 = x1 + x2 + x3 + x4
        @. y2 = x2 + x3 + x4 + x5
        @. y3 = x3 + x4 + x5 + x6
        @. y4 = x4 + x5 + x6 + x7
        @. y5 = x5 + x6 + x7 + x8
        @. y6 = x6 + x7 + x8 + x9
        @. y7 = x7 + x8 + x9 + x10
    end
end

@static get(ENV, "USE_CUDA", nothing) == "true" && using CUDA
use_cuda = @isdefined(CUDA) && CUDA.has_cuda() # will be true if you first run `using CUDA`
AType = use_cuda ? CUDA.CuArray : Array
device_name = use_cuda ? CUDA.name(CUDA.device()) : "CPU"
bm = Benchmark(; device_name, float_type = Float32)
problem_size = (50, 5, 5, 6, 50)

array_size = problem_size # array
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(;
    fused! = perf_kernel_shared_reads_fused!,
    unfused! = perf_kernel_shared_reads_unfused!,
    X,
    Y,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_unfused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_fused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)

array_size = (prod(problem_size),) # vector
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(;
    fused! = perf_kernel_shared_reads_fused!,
    unfused! = perf_kernel_shared_reads_unfused!,
    X,
    Y,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_unfused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_fused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)

tabulate_benchmark(bm)

nothing
