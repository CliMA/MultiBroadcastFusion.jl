
function materialize_args(expr::Expr)
    @assert expr.head == :call
    @assert expr.args[1] == :(Base.materialize!)
    return (expr.args[2], expr.args[3])
end

macro fused_pairs(expr)
    esc(fused_pairs(expr))
end

function _fused_pairs(expr::Expr)
    # @assert expr.head == :block
    exprs_out = []
    for _expr in expr.args
        # TODO: should we retain LineNumberNode?
        # if _expr isa Symbol # ????????
        #     error("???")
        #     return ""
        # end
        if _expr isa QuoteNode
            error("Dangling symbols are not allowed inside fused blocks")
        end
        _expr isa LineNumberNode && continue
        if _expr.head == :for
            error("Loops are not allowed inside fused blocks")
        elseif _expr.head == :if
            error("If-statements are not allowed inside fused blocks")
        elseif _expr.head == :call
            error("Function calls are not allowed inside fused blocks")
        elseif _expr.head == :(=)
            error(
                "Non-broadcast assignments are not allowed inside fused blocks",
            )
        elseif _expr.head == :let
            error("Let-blocks are not allowed inside fused blocks")
        elseif _expr.head == :quote
            error("Quotes are not allowed inside fused blocks")
        end
        if _expr.head == :macrocall && _expr.args[1] == Symbol("@__dot__")
            se = code_lowered_single_expression(_expr)
            margs = materialize_args(se)
            push!(exprs_out, :(Pair($(margs[1]), $(margs[2]))))
        else
            @show dump(_expr)
            error("Uncaught edge case")
        end
    end
    if length(exprs_out) == 1
        return "($(exprs_out[1]),)"
    else
        return "(" * join(exprs_out, ",") * ")"
    end
end

fused_pairs(expr::Expr) = Meta.parse(_fused_pairs(expr))

# General case: do nothing (identity)
substitute(x, code) = x
substitute(x::Core.SSAValue, code) = substitute(code[x.id], code)
substitute(x::Core.ReturnNode, code) = substitute(code[x.val.id], code)
substitute(s::Symbol, code) = s
# Expression: recursively substitute for Expr
substitute(e::Expr, code) =
    Expr(substitute(e.head, code), substitute.(e.args, Ref(code))...)

code_info(expr) = Base.Meta.lower(Main, expr).args[1]
function code_lowered_single_expression(expr)
    code = code_info(expr).code # vector
    s = string(substitute(code[end], code))
    return Base.Meta.parse(s)
end
