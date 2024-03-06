#=
using Revise; include(joinpath("test", "complex_example.jl"))
=#
import Base.Broadcast: broadcasted, instantiate
import BenchmarkTools
import MultiBroadcastFusion as MBF

struct CPU end
struct GPU end
device(x::Array) = CPU()
device(x) = GPU()

function copyto_cpu!(pairs::T, x::X) where {T, X}
    @inbounds for i in eachindex(x)
        knl_rcopyto!(pairs, i)
    end
    return nothing
end

Base.@propagate_inbounds function knl_rcopyto!(pair::P, idx::Int) where {P}
    dest,src = pair.first, pair.second
    @inbounds dest[idx] = src[idx]
    return nothing
end
Base.@propagate_inbounds function knl_rcopyto!(pairs::Tuple, idx::Int)
    knl_rcopyto!(first(pairs), idx)
    knl_rcopyto!(Base.tail(pairs), idx)
end
Base.@propagate_inbounds knl_rcopyto!(pairs::Tuple{T}, idx::Int) where T =
    knl_rcopyto!(first(pairs), idx)
@inline knl_rcopyto!(pairs::Tuple{}, idx::Int) = nothing

function get_array(AType, s)
    return AType(zeros(s...))
end

function get_arrays(sym, s, AType, n=10)
    fn = ntuple(i->Symbol(sym, i), n)
    return (;zip(fn,ntuple(_->get_array(AType, s), n))...)
end

# =========================================== CUDA-specific block
import CUDA
import Adapt
function copyto_cuda!(pairs::Tuple) # (Pair(dest1, bc1),Pair(dest2, bc2),...)
    nitems = length(parent(first(pairs).first))
    max_threads = 256 # can be higher if conditions permit
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    CUDA.@cuda threads = (nthreads) blocks = (nblocks) knl_multi_copyto!(pairs)
    return nothing
end
function knl_multi_copyto!(pairs::NTuple{N,T}) where {N, T<: Pair}
    nitems = length(first(pairs).first)
    idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if idx < nitems
        knl_rcopyto!(pairs, idx)
    end
    return nothing
end

function perf_kernel_hard_coded!(args, ::GPU)
    (; X, Y) = args
    x1 = X.x1
    nitems = length(parent(x1))
    max_threads = 256 # can be higher if conditions permit
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    CUDA.@cuda threads = (nthreads) blocks = (nblocks) knl_multi_copyto_hard_coded!(args, Val(nitems))
end
function knl_multi_copyto_hard_coded!(args, ::Val{nitems}) where {nitems}
    (; X, Y) = args
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    gidx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gidx < nitems
        idx = gidx
        y1[idx] = x1[idx]+x2[idx]+x3[idx]+x4[idx]
        y2[idx] = x2[idx]+x3[idx]+x4[idx]+x5[idx]
        y3[idx] = x3[idx]+x4[idx]+x5[idx]+x6[idx]
        y4[idx] = x4[idx]+x5[idx]+x6[idx]+x7[idx]
        y5[idx] = x5[idx]+x6[idx]+x7[idx]+x8[idx]
        y6[idx] = x6[idx]+x7[idx]+x8[idx]+x9[idx]
        y7[idx] = x7[idx]+x8[idx]+x9[idx]+x10[idx]
    end
    return nothing
end

@inline function adapt_pairs(pairs, ::GPU)
    to = CUDA.KernelAdaptor()
    return map(pairs) do p
        Pair(Adapt.adapt(to, p.first), Adapt.adapt(to, p.second))
    end
end
benchmark_kernel!(f!, args, ::GPU) = CUDA.@sync BenchmarkTools.@benchmark $f!($args);
benchmark_kernel!(f!, args, ::GPU) = CUDA.@sync BenchmarkTools.@benchmark $f!($args);
# ===========================================

has_cuda = CUDA.has_cuda()
AType = has_cuda ? CUDA.CuArray : Array
@show AType
arr_size = (prod((50,5,5,6,50)),)
X = get_arrays(:x, arr_size, AType)
Y = get_arrays(:y, arr_size, AType)

@inline adapt_pairs(pairs, ::CPU) = pairs

@inline adapt_pairs(pairs) = adapt_pairs(pairs, device(first(pairs).first))

function perf_kernel_unfused!(args)
    (; X, Y) = args
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    # 7 writes; 10 unique reads
    # 7 writes; 28 reads including redundant ones
    @. y1 = x1+x2+x3+x4
    @. y2 = x2+x3+x4+x5
    @. y3 = x3+x4+x5+x6
    @. y4 = x4+x5+x6+x7
    @. y5 = x5+x6+x7+x8
    @. y6 = x6+x7+x8+x9
    @. y7 = x7+x8+x9+x10
    return nothing
end
perf_kernel_fused!(args) = perf_kernel_fused!(args, device(first(args).x1))

function perf_kernel_fused!(args, ::GPU)
    (; X, Y) = args
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X;
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y;
    pairs = (
        Pair(y1, instantiate(broadcasted(+, x1, x2, x3, x4))),
        Pair(y2, instantiate(broadcasted(+, x2, x3, x4, x5))),
        Pair(y3, instantiate(broadcasted(+, x3, x4, x5, x6))),
        Pair(y4, instantiate(broadcasted(+, x4, x5, x6, x7))),
        Pair(y5, instantiate(broadcasted(+, x5, x6, x7, x8))),
        Pair(y6, instantiate(broadcasted(+, x6, x7, x8, x9))),
        Pair(y7, instantiate(broadcasted(+, x7, x8, x9, x10))),
    );
    new_pairs = adapt_pairs(pairs)
    copyto_cuda!(new_pairs)
end

function perf_kernel_fused!(args, ::CPU)
    (; X, Y) = args
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X;
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y;
    pairs = (
        Pair(y1, instantiate(broadcasted(+, x1, x2, x3, x4))),
        Pair(y2, instantiate(broadcasted(+, x2, x3, x4, x5))),
        Pair(y3, instantiate(broadcasted(+, x3, x4, x5, x6))),
        Pair(y4, instantiate(broadcasted(+, x4, x5, x6, x7))),
        Pair(y5, instantiate(broadcasted(+, x5, x6, x7, x8))),
        Pair(y6, instantiate(broadcasted(+, x6, x7, x8, x9))),
        Pair(y7, instantiate(broadcasted(+, x7, x8, x9, x10))),
    );
    new_pairs = adapt_pairs(pairs)
    copyto_cpu!(pairs, first(pairs).first)
end

perf_kernel_hard_coded!(args) = perf_kernel_hard_coded!(args, device(X.x1))

function perf_kernel_hard_coded!(args, ::CPU)
    (; X, Y) = args
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y
    @inbounds for i in eachindex(x1)
        y1[i] = x1[i]+x2[i]+x3[i]+x4[i]
        y2[i] = x2[i]+x3[i]+x4[i]+x5[i]
        y3[i] = x3[i]+x4[i]+x5[i]+x6[i]
        y4[i] = x4[i]+x5[i]+x6[i]+x7[i]
        y5[i] = x5[i]+x6[i]+x7[i]+x8[i]
        y6[i] = x6[i]+x7[i]+x8[i]+x9[i]
        y7[i] = x7[i]+x8[i]+x9[i]+x10[i]
    end
end

function benchmark_kernel!(f!, args)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = benchmark_kernel!(f!, args, device(first(args).x1))
    show(stdout, MIME("text/plain"), trial);
end

benchmark_kernel!(f!, args, ::CPU) = BenchmarkTools.@benchmark $f!($args);

# Compile
perf_kernel_unfused!((;X, Y))
perf_kernel_fused!((;X, Y))
perf_kernel_hard_coded!((;X, Y))

struct ModelA end
struct ModelB end

function perf_kernel_syntax!(args)
    (; X, Y, model) = args;
    (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X;
    (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y;
    pairs = MBF.@fused_pairs begin
        @. y1 = x1+x2+x3+x4
        @. y2 = x2+x3+x4+x5
        @. y3 = x3+x4+x5+x6
        if model isa ModelA
            @. y4 = x4+x5+x6+x7
        end
        @. y5 = x5+x6+x7+x8
        @. y6 = x6+x7+x8+x9
        @. y7 = x7+x8+x9+x10
    end;
    @test length(pairs) == 6
    new_pairs = adapt_pairs(pairs)
    if device(x1) isa GPU
        copyto_cuda!(new_pairs)
    else
        copyto_cpu!(new_pairs, first(new_pairs).first)
    end
end

model = ModelA()
# Compile
args = (;X, Y, model);
perf_kernel_syntax!(args)

# Benchmark
benchmark_kernel!(perf_kernel_syntax!, args)

nothing
