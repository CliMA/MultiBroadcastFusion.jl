#=
using Revise; include(joinpath("test", "fused.jl"))
=#

include("utils.jl")

function perf_kernel_shared_reads_unfused!(X, Y)
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    @. y1 = x1+x2+x3+x4
    @. y2 = x2+x3+x4+x5
    @. y3 = x3+x4+x5+x6
    @. y4 = x4+x5+x6+x7
    @. y5 = x5+x6+x7+x8
    @. y6 = x6+x7+x8+x9
    @. y7 = x7+x8+x9+x10
end

function perf_kernel_shared_reads_fused!(X, Y)
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    MBF.@fused begin
        @. y1 = x1+x2+x3+x4
        @. y2 = x2+x3+x4+x5
        @. y3 = x3+x4+x5+x6
        @. y4 = x4+x5+x6+x7
        @. y5 = x5+x6+x7+x8
        @. y6 = x6+x7+x8+x9
        @. y7 = x7+x8+x9+x10
    end
end

has_cuda = CUDA.has_cuda()
AType = has_cuda ? CUDA.CuArray : Array
arr_size = (prod((50,5,5,6,50)),)
X = get_arrays(:x, arr_size, AType)
Y = get_arrays(:y, arr_size, AType)

test_kernel!(;
    fused! = perf_kernel_shared_reads_fused!,
    unfused! = perf_kernel_shared_reads_unfused!,
    X, Y)
# Compile
perf_kernel_shared_reads_unfused!(X, Y)
perf_kernel_shared_reads_fused!(X, Y)

# Benchmark
benchmark_kernel!(perf_kernel_shared_reads_unfused!, X, Y)
benchmark_kernel!(perf_kernel_shared_reads_fused!, X, Y)

nothing
