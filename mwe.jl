#=
julia
include("mwe.jl")
=#
using LazyBroadcast: @lazy_broadcasted
struct Fusion{T}; pairs::T; end
import UnrolledUtilities as UU
import BenchmarkTools as BM
import Base.Broadcast.broadcasted as bcd
s = (50, 5, 5, 6, 50);
benchmark_kernel!(f!, X, Y) = show(stdout, MIME("text/plain"), BM.@benchmark($f!($X, $Y)))
arr(γ,s,T,n=10) = (; zip(ntuple(i -> Symbol(γ, i), n), ntuple(_ -> T(zeros(s...)), n))...)
function fused_copyto_manually_inlined!(fmb::Fusion)
    dests = map(x -> x.first, fmb.pairs)
    ei = eltype(dests) <: Vector ? eachindex(dests...) :
        eachindex(IndexCartesian(), dests...)
    @inbounds @simd ivdep for i in ei
        fmb.pairs[1].first[i] = fmb.pairs[1].second[i]
        fmb.pairs[2].first[i] = fmb.pairs[2].second[i]
    end
end

function fused_copyto_counter_intuitive_but_noinline!(fmb::Fusion)
    dests = map(x -> x.first, fmb.pairs)
    ei = eltype(dests) <: Vector ? eachindex(dests...) :
        eachindex(IndexCartesian(), dests...)
    @inbounds for i in ei
        for (dest, bc) in fmb.pairs
            dest[i] = bc[i]
        end
    end
end

function fused_copyto_base!(fmb::Fusion)
    dests = map(x -> x.first, fmb.pairs)
    ei = eltype(dests) <: Vector ? eachindex(dests...) :
        eachindex(IndexCartesian(), dests...)
    for (dest, bc) in fmb.pairs
        @inbounds @simd ivdep for i in ei
            dest[i] = bc[i]
        end
    end
end

function kernel_fused!(X, Y)
    (;x1, x2, x3, x4) = X; (;y1, y2) = Y;
    bc1 = bcd(+, bcd(*, x1, x2), bcd(*, x3, x4))
    bc2 = bcd(+, bcd(*, x1, x3), bcd(*, x2, x4))
    fused_copyto!(Fusion((Pair(y1, bc1), Pair(y2, bc2))))
end
function kernel_unfused!(X, Y)
    (;x1, x2, x3, x4) = X; (;y1, y2) = Y;
    @. y1 = x1 * x2 + x3 * x4
    @. y2 = x1 * x3 + x2 * x4
end
X = arr(:x, s, Array); Y = arr(:y, s, Array); # data, X.x1, X.x2, ..., Y.y1, Y.y2
benchmark_kernel!(kernel_fused!, X, Y)
benchmark_kernel!(kernel_unfused!, X, Y)
