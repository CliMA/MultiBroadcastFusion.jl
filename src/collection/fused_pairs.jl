#####
##### Simple version
#####

# General case: do nothing (identity)
transform(x) = x
transform(s::Symbol) = s
# Expression: recursively transform for Expr
function transform(e::Expr)
    if e.head == :macrocall && e.args[1] == Symbol("@__dot__")
        se = code_lowered_single_expression(e)
        margs = materialize_args(se)
        subexpr = :(Pair($(margs[1]), $(margs[2])))
        subexpr
    else
        Expr(transform(e.head), transform.(e.args)...)
    end
end

"""
    fused_pairs

Function that fuses broadcast expressions that
are immediately one after another. For example:

```julia
import MultiBroadcastFusion as MBF
MBF.@make_type MyFusedMultiBroadcast
MBF.@make_fused fused_pairs MyFusedMultiBroadcast fused
```

To use `MultiBroadcastFusion`'s `@fused` macro:
```
import MultiBroadcastFusion as MBF
x = rand(1);y = rand(1);z = rand(1);
MBF.@fused begin
    @. x += y
    @. z += y
end
```
"""
function fused_pairs(expr::Expr)
    check_restrictions(expr)
    e = transform(expr)
    @assert e.head == :block
    ex = Expr(:call, :tuple, e.args...)
    # Filter out LineNumberNode, as this will not be valid due to prepending `tup = ()`
    linefilter!(ex)
    ex
end

function check_restrictions(expr::Expr)
    for _expr in expr.args
        _expr isa LineNumberNode && continue
        s_error = if _expr isa QuoteNode
            "Dangling symbols are not allowed inside fused blocks"
        elseif _expr.head == :for
            "Loops are not allowed inside fused blocks"
        elseif _expr.head == :if
            "If-statements are not allowed inside fused blocks"
        elseif _expr.head == :call
            "Function calls are not allowed inside fused blocks"
        elseif _expr.head == :(=)
            "Non-broadcast assignments are not allowed inside fused blocks"
        elseif _expr.head == :let
            "Let-blocks are not allowed inside fused blocks"
        elseif _expr.head == :quote
            "Quotes are not allowed inside fused blocks"
        else
            ""
        end
        isempty(s_error) || error(s_error)
        if _expr.head == :macrocall && _expr.args[1] == Symbol("@__dot__")
        else
            @show dump(_expr)
            error("Uncaught edge case")
        end
    end
end
