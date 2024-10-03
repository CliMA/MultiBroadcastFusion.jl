#=
using Revise; include(joinpath("test", "execution", "bm_fused_shared_reads_writes.jl"))
=#

include("utils_test.jl")
include("utils_setup.jl")
include("utils_benchmark.jl")

import MultiBroadcastFusion as MBF

function perf_kernel_shared_reads_writes_unfused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    # Totoal: 10 writes, 15 reads, and 5 read/write overlaps
    @. y1 = x1 + x6
    @. y2 = x2 + x7
    @. y3 = x3 + x8
    @. y4 = x4 + x9
    @. y5 = x5 + x10
    @. y6 = y1
    @. y7 = y2
    @. y8 = y3
    @. y9 = y4
    @. y10 = y5
end

function perf_kernel_shared_reads_writes_fused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    MBF.@fused_direct begin
        @. y1 = x1 + x6
        @. y2 = x2 + x7
        @. y3 = x3 + x8
        @. y4 = x4 + x9
        @. y5 = x5 + x10
        @. y6 = y1
        @. y7 = y2
        @. y8 = y3
        @. y9 = y4
        @. y10 = y5
    end
end

@static get(ENV, "USE_CUDA", nothing) == "true" && using CUDA
use_cuda = @isdefined(CUDA) && CUDA.has_cuda() # will be true if you first run `using CUDA`
AType = use_cuda ? CUDA.CuArray : Array
device_name = use_cuda ? CUDA.name(CUDA.device()) : "CPU"
bm = Benchmark(;
    problem_size = (prod((50, 5, 5, 6, 50)),),
    device_name,
    float_type = Float32,
)
problem_size = (50, 5, 5, 6, 50)

array_size = problem_size # array
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(;
    unfused! = perf_kernel_shared_reads_writes_unfused!,
    fused! = perf_kernel_shared_reads_writes_fused!,
    X,
    Y,
)
# Benchmark
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_writes_unfused!,
    X,
    Y;
    n_reads_writes = 10 + 15,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_writes_fused!,
    X,
    Y;
    n_reads_writes = 10 + 15,
    problem_size = array_size,
)

array_size = (prod(problem_size),) # vector
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(;
    unfused! = perf_kernel_shared_reads_writes_unfused!,
    fused! = perf_kernel_shared_reads_writes_fused!,
    X,
    Y,
)
# Benchmark
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_writes_unfused!,
    X,
    Y;
    n_reads_writes = 10 + 15,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_shared_reads_writes_fused!,
    X,
    Y;
    n_reads_writes = 10 + 15,
    problem_size = array_size,
)


tabulate_benchmark(bm)

nothing
