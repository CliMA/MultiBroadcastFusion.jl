module MultiBroadcastFusion


abstract type AbstractFusedMultiBroadcast end

"""
	FusedMultiBroadcast(pairs::Tuple)

A mult-broadcast fusion object
"""
struct FusedMultiBroadcast{T} <: AbstractFusedMultiBroadcast
    pairs::T
end

# Base.@propagate_inbounds function rcopyto_at!(pair::Pair, i::CartesianIndex)
#     dest,src = pair.first, pair.second
#     @inbounds src_i = src[i]
#     @inbounds dest[i] = src_i
#     return nothing
# end
# Base.@propagate_inbounds function rcopyto_at!(pair::Pair, i::Int)
#     dest,src = pair.first, pair.second
#     @inbounds src_i = src[i]
#     @inbounds dest[i] = src_i
#     return nothing
# end
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

include("macro_utils.jl")

end # module MultiBroadcastFusion
