#####
##### Helper
#####

# Recursively remove LineNumberNode from an `Expr`
@noinline function linefilter!(expr::Expr)
    total = length(expr.args)
    i = 0
    while i < total
        i += 1
        if expr.args[i] |> typeof == Expr
            if expr.args[i].head == :line
                deleteat!(expr.args, i)
                total -= 1
                i -= 1
            else
                expr.args[i] = linefilter!(expr.args[i])
            end
        elseif expr.args[i] |> typeof == LineNumberNode
            if expr.head == :macrocall
                expr.args[i] = nothing
            else
                deleteat!(expr.args, i)
                total -= 1
                i -= 1
            end
        end
    end
    return expr
end

function materialize_args(expr::Expr)
    @assert expr.head == :call
    if expr.args[1] == :(Base.materialize!)
        return (expr.args[2], expr.args[3])
    elseif expr.args[1] == :(Base.materialize)
        return (expr.args[2], expr.args[2])
    else
        error("Uncaught edge case.")
    end
end

const dot_ops = (
    Symbol(".+"),
    Symbol(".-"),
    Symbol(".*"),
    Symbol("./"),
    Symbol(".="),
    Symbol(".=="),
    Symbol(".≠"),
    Symbol(".^"),
    Symbol(".!="),
    Symbol(".>"),
    Symbol(".<"),
    Symbol(".>="),
    Symbol(".<="),
    Symbol(".≤"),
    Symbol(".≥"),
)
isa_dot_op(op) = any(x -> op == x, dot_ops)
