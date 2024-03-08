module MultiBroadcastFusion

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

"""
    @make_fused type_name fused_named

This macro
 - Imports MultiBroadcastFusion
 - Defines a type, `type_name`
 - Defines a macro, `@fused_name`

This allows users to flexibility
to customize their broadcast fusion.

# Example
```julia
import MultiBroadcastFusion as MBF
MBF.@make_fused MyFusedBroadcast my_fused

Base.copyto!(fmb::MyFusedBroadcast) = println("You're ready to fuse!")

x1 = rand(3,3)
y1 = rand(3,3)
y2 = rand(3,3)

# 4 reads, 2 writes
@my_fused begin
  @. y1 = x1
  @. y2 = x1
end
```
"""
macro make_fused(type_name, fused_name)
    t = esc(type_name)
    f = esc(fused_name)
    return quote
        struct $t{T <: Tuple}
            pairs::T
        end
        macro $f(expr)
            _pairs = esc($(fused_pairs)(expr))
            t = $t
            quote
                Base.copyto!($t($_pairs))
            end
        end
    end
end

end # module MultiBroadcastFusion
