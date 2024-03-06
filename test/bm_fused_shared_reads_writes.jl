#=
using Revise; include(joinpath("test", "fused_reads_writes.jl"))
=#

include("utils.jl")

function perf_kernel_shared_reads_writes_unfused!(X, Y)
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    # Totoal: 10 writes, 15 reads, and 5 read/write overlaps
    @. y1 = x1+x6
    @. y2 = x2+x7
    @. y3 = x3+x8
    @. y4 = x4+x9
    @. y5 = x5+x10
    @. y6 = y1
    @. y7 = y2
    @. y8 = y3
    @. y9 = y4
    @. y10 = y5
end

function perf_kernel_shared_reads_writes_fused!(X, Y)
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X;
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y;
    MBF.@fused begin
        @. y1 = x1+x6
        @. y2 = x2+x7
        @. y3 = x3+x8
        @. y4 = x4+x9
        @. y5 = x5+x10
        @. y6 = y1
        @. y7 = y2
        @. y8 = y3
        @. y9 = y4
        @. y10 = y5
    end
end

has_cuda = CUDA.has_cuda()
AType = has_cuda ? CUDA.CuArray : Array
arr_size = (prod((50,5,5,6,50)),)
X = get_arrays(:x, arr_size, AType)
Y = get_arrays(:y, arr_size, AType)

test_kernel!(;
    unfused! = perf_kernel_shared_reads_writes_unfused!,
    fused! = perf_kernel_shared_reads_writes_fused!,
    X, Y)
# Compile
perf_kernel_shared_reads_writes_unfused!(X, Y)
perf_kernel_shared_reads_writes_fused!(X, Y)

# Benchmark
benchmark_kernel!(perf_kernel_shared_reads_writes_unfused!, X, Y)
benchmark_kernel!(perf_kernel_shared_reads_writes_fused!, X, Y)

nothing
