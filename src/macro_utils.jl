
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
        if _expr isa Symbol # ????????
            return ""
        end
        _expr isa LineNumberNode && continue
        # @assert _expr isa Expr
        if _expr.head == :macrocall && _expr.args[1] == Symbol("@__dot__")
            se = code_lowered_single_expression(_expr)
            margs = materialize_args(se)
            push!(exprs_out, :(Pair($(margs[1]), $(margs[2]))))
        end
    end
    if length(exprs_out) == 1
        return "($(exprs_out[1]),)"
    else
        return "(" * join(exprs_out, ",") * ")"
    end
end

fused_pairs(expr::Expr) = Meta.parse(_fused_pairs(expr))

function build_expr(s::String, code_remain)
    n_subs = count("%", s)
    if n_subs > 0
        while n_subs > 0
            regex = r"%[0-9]"
            m = match(regex, s)
            smatch = m.match
            j = Meta.parse(smatch[2:end])
            s = replace(s, smatch => string(code_remain[j]))
            n_subs = count("%", s)
        end
    else
        return s
    end
    return s
end

build_expr(code::Vector) = build_expr(string(code[end]), code)

function code_lowered_single_expression(expr)
    code_lowered = Base.Meta.lower(Main, expr)
    code_info = code_lowered.args[1]
    code = code_info.code # vector
    s = build_expr(code)
    if startswith(s, "return ")
        s = replace(s, "return " => "")
    end
    return Base.Meta.parse(s)
end
