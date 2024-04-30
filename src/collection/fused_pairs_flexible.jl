#####
##### Complex/flexible version
#####

# General case: do nothing (identity)
transform_flex(x, sym) = x
transform_flex(s::Symbol, sym) = s
# Expression: recursively transform_flex for Expr
function transform_flex(e::Expr, sym)
    if e.head == :macrocall && e.args[1] == Symbol("@__dot__")
        se = code_lowered_single_expression(e)
        margs = materialize_args(se)
        subexpr = :($sym = ($sym..., Pair($(margs[1]), $(margs[2]))))
        subexpr
    else
        Expr(transform_flex(e.head, sym), transform_flex.(e.args, sym)...)
    end
end

"""
    fused_pairs_flexible

Function that fuses broadcast expressions
that stride flow control logic. For example:

```julia
import MultiBroadcastFusion as MBF
MBF.@make_type MyFusedMultiBroadcast
MBF.@make_fused fused_pairs_flexible MyFusedMultiBroadcast fused_flexible
```

To use `MultiBroadcastFusion`'s `@fused_flexible` macro:
```
import MultiBroadcastFusion as MBF
x = rand(1);y = rand(1);z = rand(1);
MBF.@fused_flexible begin
    @. x += y
    @. z += y
end
```
"""
function fused_pairs_flexible(expr::Expr, sym::Symbol)
    check_restrictions_flexible(expr)
    e = transform_flex(expr, sym)
    @assert e.head == :block
    ex = Expr(:block, :($sym = ()), e.args..., sym)
    # Filter out LineNumberNode, as this will not be valid due to prepending `tup = ()`
    linefilter!(ex)
    ex
end

function check_restrictions_flexible(expr::Expr)
    for arg in expr.args
        arg isa LineNumberNode && continue
        s_error = if arg isa QuoteNode
            "Dangling symbols are not allowed inside fused blocks"
        elseif arg.head == :call
            "Function calls are not allowed inside fused blocks"
        elseif arg.head == :(=)
            "Non-broadcast assignments are not allowed inside fused blocks"
        elseif arg.head == :let
            "Let-blocks are not allowed inside fused blocks"
        elseif arg.head == :quote
            "Quotes are not allowed inside fused blocks"
        else
            ""
        end
        isempty(s_error) || error(s_error)

        if arg.head == :macrocall && arg.args[1] == Symbol("@__dot__")
        elseif arg.head == :for
            check_restrictions(arg.args[2])
        elseif arg.head == :if
            check_restrictions(arg.args[2])
        elseif arg.head == :macrocall && arg.args[1] == Symbol("@inbounds")
        else
            @show dump(arg)
            error("Uncaught edge case")
        end
    end
    return nothing
end
