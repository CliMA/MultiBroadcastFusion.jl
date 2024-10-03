#=
using Revise; include(joinpath("test", "execution", "bm_fused_reads_vs_hard_coded.jl"))
=#
include("utils_test.jl")
include("utils_setup.jl")
include("utils_benchmark.jl")

import MultiBroadcastFusion as MBF

# =========================================== hard-coded implementations
perf_kernel_hard_coded!(X, Y) = perf_kernel_hard_coded!(X, Y, MBF.device(X.x1))

function perf_kernel_hard_coded!(X, Y, ::MBF.CPU)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    @inbounds for i in eachindex(x1)
        y1[i] = x1[i] + x2[i] + x3[i] + x4[i]
        y2[i] = x2[i] + x3[i] + x4[i] + x5[i]
        y3[i] = x3[i] + x4[i] + x5[i] + x6[i]
        y4[i] = x4[i] + x5[i] + x6[i] + x7[i]
        y5[i] = x5[i] + x6[i] + x7[i] + x8[i]
        y6[i] = x6[i] + x7[i] + x8[i] + x9[i]
        y7[i] = x7[i] + x8[i] + x9[i] + x10[i]
    end
end

@static get(ENV, "USE_CUDA", nothing) == "true" && using CUDA
use_cuda = @isdefined(CUDA) && CUDA.has_cuda() # will be true if you first run `using CUDA`
@static if use_cuda
    function perf_kernel_hard_coded!(X, Y, ::MBF.GPU)
        x1 = X.x1
        nitems = length(parent(x1))
        max_threads = 256 # can be higher if conditions permit
        nthreads = min(max_threads, nitems)
        nblocks = cld(nitems, nthreads)
        CUDA.@cuda threads = (nthreads) blocks = (nblocks) knl_multi_copyto_hard_coded!(
            X,
            Y,
            Val(nitems),
        )
    end
    function knl_multi_copyto_hard_coded!(X, Y, ::Val{nitems}) where {nitems}
        (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
        (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
        idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
        @inbounds begin
            if idx ≤ nitems
                y1[idx] = x1[idx] + x2[idx] + x3[idx] + x4[idx]
                y2[idx] = x2[idx] + x3[idx] + x4[idx] + x5[idx]
                y3[idx] = x3[idx] + x4[idx] + x5[idx] + x6[idx]
                y4[idx] = x4[idx] + x5[idx] + x6[idx] + x7[idx]
                y5[idx] = x5[idx] + x6[idx] + x7[idx] + x8[idx]
                y6[idx] = x6[idx] + x7[idx] + x8[idx] + x9[idx]
                y7[idx] = x7[idx] + x8[idx] + x9[idx] + x10[idx]
            end
        end
        return nothing
    end
end

# ===========================================

function perf_kernel_unfused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    # 7 writes; 10 unique reads
    # 7 writes; 28 reads including redundant ones
    @. y1 = x1 + x2 + x3 + x4
    @. y2 = x2 + x3 + x4 + x5
    @. y3 = x3 + x4 + x5 + x6
    @. y4 = x4 + x5 + x6 + x7
    @. y5 = x5 + x6 + x7 + x8
    @. y6 = x6 + x7 + x8 + x9
    @. y7 = x7 + x8 + x9 + x10
    return nothing
end

function perf_kernel_fused!(X, Y)
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
    fused! = perf_kernel_fused!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)
use_cuda && test_kernel!(;
    fused! = perf_kernel_hard_coded!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_unfused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_fused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
use_cuda && push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_hard_coded!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)

array_size = (prod(problem_size),) # vector
X = get_arrays(:x, AType, bm.float_type, array_size)
Y = get_arrays(:y, AType, bm.float_type, array_size)
test_kernel!(;
    fused! = perf_kernel_fused!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)
use_cuda && test_kernel!(;
    fused! = perf_kernel_hard_coded!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_unfused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_fused!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)
use_cuda && push_benchmark!(
    bm,
    use_cuda,
    perf_kernel_hard_coded!,
    X,
    Y;
    n_reads_writes = 7 + 10,
    problem_size = array_size,
)


tabulate_benchmark(bm)

nothing
