#=
using Revise; include(joinpath("test", "bm_fused_reads_vs_hard_coded.jl"))
=#
include("utils.jl")

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

# ===========================================

has_cuda = CUDA.has_cuda()
AType = has_cuda ? CUDA.CuArray : Array
arr_size = (prod((50, 5, 5, 6, 50)),)
# arr_size = (50,5,5,6,50)
X = get_arrays(:x, arr_size, AType);
Y = get_arrays(:y, arr_size, AType);

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
    MBF.@fused begin
        @. y1 = x1 + x2 + x3 + x4
        @. y2 = x2 + x3 + x4 + x5
        @. y3 = x3 + x4 + x5 + x6
        @. y4 = x4 + x5 + x6 + x7
        @. y5 = x5 + x6 + x7 + x8
        @. y6 = x6 + x7 + x8 + x9
        @. y7 = x7 + x8 + x9 + x10
    end
end

test_kernel!(;
    fused! = perf_kernel_fused!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)
test_kernel!(;
    fused! = perf_kernel_hard_coded!,
    unfused! = perf_kernel_unfused!,
    X,
    Y,
)

# Compile
perf_kernel_unfused!(X, Y)
perf_kernel_fused!(X, Y)
perf_kernel_hard_coded!(X, Y)

# Benchmark
benchmark_kernel!(perf_kernel_unfused!, X, Y)
benchmark_kernel!(perf_kernel_fused!, X, Y)
benchmark_kernel!(perf_kernel_hard_coded!, X, Y)

nothing
