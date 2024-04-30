"""
    @make_type type_name

This macro defines a type `type_name`, to be
passed to `@make_fused`.
"""
macro make_type(type_name)
    t = esc(type_name)
    return quote
        struct $t{T <: Union{Tuple, AbstractArray}}
            pairs::T
        end
    end
end

"""
    @make_fused fusion_type type_name fused_named

This macro
 - Defines a type type_name
 - Defines a macro, `@fused_name`, using the fusion type `fusion_type`

This allows users to flexibility
to customize their broadcast fusion.

# Example
```julia
import MultiBroadcastFusion as MBF
MBF.@make_type MyFusedBroadcast
MBF.@make_fused MBF.fused_pairs MyFusedBroadcast my_fused

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
macro make_fused(fusion_type, type_name, fused_name)
    t = esc(type_name)
    f = esc(fused_name)
    return quote
        macro $f(expr)
            _pairs = esc($(fusion_type)(expr))
            t = $t
            quote
                Base.copyto!($t($_pairs))
            end
        end
    end
end
