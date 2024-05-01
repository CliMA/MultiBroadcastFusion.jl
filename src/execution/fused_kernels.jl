@make_type FusedMultiBroadcast
@make_fused fused_direct FusedMultiBroadcast fused_direct
@make_fused fused_assemble FusedMultiBroadcast fused_assemble

struct CPU end
struct GPU end
device(x::AbstractArray) = CPU()

function Base.copyto!(fmb::FusedMultiBroadcast)
    pairs = fmb.pairs # (Pair(dest1, bc1),Pair(dest2, bc2),...)
    dest = first(pairs).first
    fused_copyto!(fmb, device(dest))
end

Base.@propagate_inbounds function rcopyto_at!(pair::Pair, i...)
    dest, src = pair.first, pair.second
    @inbounds dest[i...] = src[i...]
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, i...)
    rcopyto_at!(first(pairs), i...)
    rcopyto_at!(Base.tail(pairs), i...)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, i...) =
    rcopyto_at!(first(pairs), i...)
@inline rcopyto_at!(pairs::Tuple{}, i...) = nothing

# This is better than the baseline.
function fused_copyto!(fmb::FusedMultiBroadcast, ::CPU)
    (; pairs) = fmb
    destinations = map(x -> x.first, pairs)
    ei = if eltype(destinations) <: Vector
        eachindex(destinations...)
    else
        eachindex(IndexCartesian(), destinations...)
    end
    for (dest, bc) in pairs
        @inbounds @simd ivdep for i in ei
            dest[i] = bc[i]
        end
    end
end


# This should, in theory be better, but it seems like inlining is
# failing somewhere.
# function fused_copyto!(fmb::FusedMultiBroadcast, ::CPU)
#     (; pairs) = fmb
#     destinations = map(x -> x.first, pairs)
#     ei = if eltype(destinations) <: Vector
#         eachindex(destinations...)
#     else
#         eachindex(IndexCartesian(), destinations...)
#     end
#     @inbounds @simd ivdep for i in ei
#         MBF.rcopyto_at!(pairs, i)
#     end
# end
