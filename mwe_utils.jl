materialize_args(expr::Expr) = expr.args[1] == :(Base.materialize!) ?
    (expr.args[2], expr.args[3]) : (expr.args[2], expr.args[2])
sub(x, code) = x
sub(x::Core.SSAValue, code) = sub(code[x.id], code)
sub(x::Core.ReturnNode, code) = sub(code[x.val.id], code)
sub(s::Symbol, code) = s
sub(e::Expr, code) =
    Expr(sub(e.head, code), sub.(e.args, Ref(code))...)
code_info(expr) = Base.Meta.lower(Main, expr).args[1]
function clse(expr)
    code = code_info(expr).code # vector
    return Base.Meta.parse(string(sub(code[end], code)))
end
transform(x) = x
transform(s::Symbol) = s
function transform(e::Expr)
    if e.head == :macrocall && e.args[1] == Symbol("@__dot__")
        :($(materialize_args(clse(e))[2]))
    else
        Expr(transform(e.head), transform.(e.args)...)
    end
end
macro lazy_broadcasted(expr)
    esc(transform(expr))
end
