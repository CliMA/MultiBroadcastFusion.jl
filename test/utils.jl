import Base.Broadcast: broadcasted, instantiate
import BenchmarkTools
import MultiBroadcastFusion as MBF
using Test

function get_array(AType, s)
    return AType(zeros(s...))
end

function get_arrays(sym, s, AType, n = 10)
    println("array_size = $s, array_type = $AType")
    fn = ntuple(i -> Symbol(sym, i), n)
    return (; zip(fn, ntuple(_ -> get_array(AType, s), n))...)
end

struct CPU end
struct GPU end
device(x::Array) = CPU()
device(x) = GPU()

# We're defining a global method, so we can't include this
# in every file without getting warnings. To silence the warnings,
# we've wrapped this in an if-statement.
# WARNING:
#    If you've updated `Base.copyto!(fmb::FusedMultiBroadcast)`,
#    then Revise will not update this method!!!
MBF.@make_fused FusedMultiBroadcast fused
if !hasmethod(Base.copyto!, Tuple{<:FusedMultiBroadcast})
    function Base.copyto!(fmb::FusedMultiBroadcast)
        pairs = fmb.pairs
        dest = first(pairs).first
        @assert device(dest) isa CPU || device(dest) isa GPU
        if device(dest) isa GPU
            new_pairs = adapt_pairs(pairs)
            copyto_cuda!(new_pairs)
        else
            destinations = map(x -> x.first, pairs)
            ei = if eltype(destinations) <: Vector
                eachindex(destinations...)
            else
                eachindex(IndexCartesian(), destinations...)
            end
            copyto_cpu!(pairs, ei)
        end
    end
end

# This is better than the baseline.
function copyto_cpu!(pairs::T, ei::EI) where {T, EI}
    for (dest, bc) in pairs
        @inbounds @simd ivdep for i in ei
            dest[i] = bc[i]
        end
    end
    return nothing
end

# This should, in theory be better, but it seems like inlining is
# failing somewhere.
# function copyto_cpu!(pairs::T, ei::EI) where {T, EI}
#     @inbounds @simd ivdep for i in ei
#         MBF.rcopyto_at!(pairs, i)
#     end
#     return nothing
# end

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
function knl_multi_copyto!(pairs::Tuple)
    nitems = length(first(pairs).first)
    idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if idx ≤ nitems
        MBF.rcopyto_at!(pairs, idx)
    end
    return nothing
end

@inline adapt_pairs(pairs, ::CPU) = pairs
@inline adapt_pairs(pairs) = adapt_pairs(pairs, device(first(pairs).first))
@inline function adapt_pairs(pairs, ::GPU)
    to = CUDA.KernelAdaptor()
    return map(pairs) do p
        Pair(Adapt.adapt(to, p.first), Adapt.adapt(to, p.second))
    end
end

# benchmarking
function benchmark_kernel!(f!, X, Y)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = benchmark_kernel!(f!, X, Y, device(X.x1))
    show(stdout, MIME("text/plain"), trial)
end
benchmark_kernel!(f!, X, Y, ::GPU) =
    CUDA.@sync BenchmarkTools.@benchmark $f!($X, $Y);
benchmark_kernel!(f!, X, Y, ::CPU) = BenchmarkTools.@benchmark $f!($X, $Y);

function benchmark_kernel!(f!, args)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = benchmark_kernel!(f!, args, device(first(args).x1))
    show(stdout, MIME("text/plain"), trial)
end
benchmark_kernel!(f!, args, ::GPU) =
    CUDA.@sync BenchmarkTools.@benchmark $f!($args);
benchmark_kernel!(f!, args, ::CPU) = BenchmarkTools.@benchmark $f!($args);

function show_diff(A, B)
    for pn in propertynames(A)
        Ai = getproperty(A, pn)
        Bi = getproperty(B, pn)
        @show Ai, abs.(Ai - Bi)
    end
end

function compare(A, B)
    pass = true
    for pn in propertynames(A)
        pass = pass && all(getproperty(A, pn) .== getproperty(B, pn))
    end
    pass || show_diff(A, B)
    return pass
end
function test_kernel!(; fused!, unfused!, X, Y)
    for x in X
        x .= map(_ -> rand(), x)
    end
    for y in Y
        y .= map(_ -> rand(), y)
    end
    X_fused = deepcopy(X)
    X_unfused = deepcopy(X)
    Y_fused = deepcopy(Y)
    Y_unfused = deepcopy(Y)
    fused!(X_fused, Y_fused)
    unfused!(X_unfused, Y_unfused)
    @testset "Test correctness of $(nameof(typeof(fused!)))" begin
        @test compare(X_fused, X_unfused)
        @test compare(Y_fused, Y_unfused)
    end
end
function test_kernel_args!(; fused!, unfused!, args)
    (; X, Y) = args
    for x in X
        x .= rand(size(x)...)
    end
    for y in Y
        y .= rand(size(y)...)
    end
    X_fused = deepcopy(X)
    X_unfused = deepcopy(X)
    Y_fused = deepcopy(Y)
    Y_unfused = deepcopy(Y)
    args_fused = (; X = X_fused, Y = Y_fused)
    args_unfused = (; X = X_unfused, Y = Y_unfused)
    fused!(args_fused)
    unfused!(args_unfused)
    @testset "Test correctness of $(nameof(typeof(fused!)))" begin
        @test compare(X_fused, X_unfused)
        @test compare(Y_fused, Y_unfused)
    end
end
