@make_type FusedMultiBroadcast
@make_fused fused_direct FusedMultiBroadcast fused_direct
@make_fused fused_assemble FusedMultiBroadcast fused_assemble

struct MBF_CPU end
struct MBF_CUDA end
device(x::AbstractArray) = MBF_CPU()

function Base.copyto!(fmb::FusedMultiBroadcast)
    # Since we intercept Base.copyto!, we have not yet
    # called Base.Broadcast.instantiate (as this is done
    # in materialize, which has been stripped away), so,
    # let's call it here.
    destinations = map(x -> x.first, fmb.pairs)
    bcs = map(x -> x.second, fmb.pairs)
    bcs = map(Base.Broadcast.instantiate, bcs)
    fmb′ = FusedMultiBroadcast((destinations, bcs))
    dest = first(destinations)
    fused_copyto!(fmb′, device(dest))
end

Base.@propagate_inbounds function rcopyto_at!(dest, bc, i::Vararg{T}) where {T}
    @inbounds dest[i...] = bc[i...]
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(
    dest::Tuple,
    bcs::Tuple,
    i::Vararg{T},
) where {T}
    rcopyto_at!(first(dest), first(bcs), i...)
    rcopyto_at!(Base.tail(dest), Base.tail(bcs), i...)
end
Base.@propagate_inbounds rcopyto_at!(
    dest::Tuple{<:Any},
    bcs::Tuple{<:Any},
    i::Vararg{T},
) where {T} = rcopyto_at!(first(dest), first(bcs), i...)
@inline rcopyto_at!(::Tuple{}, ::Tuple{}, i::Vararg{T}) where {T} = nothing

Base.@propagate_inbounds function rcopyto_at!(dest, bc, i::Vararg{T}) where {T}
    @inbounds dest[i...] = bc[i...]
    return nothing
end

@generated function generated_rcopyto_at!(
    destinations,
    bcs,
    i,
    ::Val{N},
) where {N}
    return quote
        Base.Cartesian.@nexprs $N ξ -> begin
            @inbounds destinations[ξ][i] = bcs[ξ][i]
        end
    end
end

@generated function revert_fusion!(destinations, bcs, ei, ::Val{N}) where {N}
    return quote
        Base.Cartesian.@nexprs $N ξ -> begin
            dest = destinations[ξ]
            bc = bcs[ξ]
            @simd for i in ei
                @inbounds dest[i] = bc[i]
            end
        end
    end
end

# This is better than the baseline.
function fused_copyto!(fmb::FusedMultiBroadcast, ::MBF_CPU)
    (; pairs) = fmb
    destinations = first(pairs)
    bcs = last(pairs)
    ei = if eltype(destinations) <: Vector
        eachindex(destinations...)
    else
        eachindex(IndexCartesian(), destinations...)
    end
    N = length(destinations)
    # generated_rcopyto_at!(destinations, bcs, i, Val(N))
    revert_fusion!(destinations, bcs, ei, Val(N))
    return destinations
end


# This should (in theory) be better but it seems like
# inlining or simd is failing somewhere.
# function fused_copyto!(fmb::FusedMultiBroadcast, ::MBF_CPU)
#     (; pairs) = fmb
#     destinations = first(pairs)
#     bcs = last(pairs)
#     ei = if eltype(destinations) <: Vector
#         eachindex(destinations...)
#     else
#         eachindex(IndexCartesian(), destinations...)
#     end
#     @inbounds @simd ivdep for i in ei
#         rcopyto_at!(destinations, bcs, i)
#     end
# end
