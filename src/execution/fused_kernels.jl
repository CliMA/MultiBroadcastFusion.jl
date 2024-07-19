@make_type FusedMultiBroadcast
@make_fused fused_direct FusedMultiBroadcast fused_direct
@make_fused fused_assemble FusedMultiBroadcast fused_assemble

import KernelAbstractions as KA
using KernelAbstractions

# For tests, we can move this out.
struct MBF_CPU end
struct MBF_CUDA end
device(x::AbstractArray) = MBF_CPU()

KA.@kernel function fused_copyto_kernel!(fmb::FusedMultiBroadcast)
    (; pairs) = fmb
    I = @index(Global, Cartesian)
    rcopyto_at!(pairs, I)
end

function Base.copyto!(fmb::FusedMultiBroadcast)
    # Since we intercept Base.copyto!, we have not yet
    # called Base.Broadcast.instantiate (as this is done
    # in materialize, which has been stripped away), so,
    # let's call it here.
    fmb′ = FusedMultiBroadcast(
        map(fmb.pairs) do p
            Pair(p.first, Base.Broadcast.instantiate(p.second))
        end,
    )
    (; pairs) = fmb′ # (Pair(dest1, bc1),Pair(dest2, bc2),...)
    dest = first(pairs).first

    assert_sizes(pairs)
    # assert_backends(pairs) # perhaps its fine to just compare all `dest` backends?
    dest1 = first(pairs).first
    backend = KA.get_backend(dest1)
    kernel = fused_copyto_kernel!(backend)
    kernel(fmb′; ndrange = size(dest1))
end

#####
##### rcopyto_at!
#####

Base.@propagate_inbounds function rcopyto_at!(pair::Pair, I)
    dest, src = pair.first, pair.second
    rcopyto_at!(dest, src, I)
    return nothing
end
# Base.@propagate_inbounds function rcopyto_at!(dest, @Const(src), I) # can't use @Const(src) here...
Base.@propagate_inbounds function rcopyto_at!(dest::AbstractVector, src, I)
    @inbounds dest[I] = src[I]
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(dest::AbstractArray, src, I)
    @inbounds dest[I] = src[I]
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, I)
    rcopyto_at!(first(pairs), I)
    rcopyto_at!(Base.tail(pairs), I)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, I) =
    rcopyto_at!(first(pairs), I)
@inline rcopyto_at!(pairs::Tuple{}, I) = nothing

#####
##### assert_sizes
#####

Base.@propagate_inbounds function assert_sizes(pair::Pair)
    dest, src = pair.first, pair.second
    @assert size(dest) == size(src)
    return nothing
end
Base.@propagate_inbounds function assert_sizes(pairs::Tuple)
    assert_sizes(first(pairs))
    assert_sizes(Base.tail(pairs))
end
Base.@propagate_inbounds assert_sizes(pairs::Tuple{<:Any}) =
    assert_sizes(first(pairs))
@inline assert_sizes(pairs::Tuple{}) = nothing

#####
##### assert_backends
#####

Base.@propagate_inbounds function assert_backends(pair::Pair)
    dest, src = pair.first, pair.second
    @assert KA.get_backend(dest) == KA.get_backend(src)
    return nothing
end
Base.@propagate_inbounds function assert_backends(pairs::Tuple)
    assert_backends(first(pairs))
    assert_backends(Base.tail(pairs))
end
Base.@propagate_inbounds assert_backends(pairs::Tuple{<:Any}) =
    assert_backends(first(pairs))
@inline assert_backends(pairs::Tuple{}) = nothing
