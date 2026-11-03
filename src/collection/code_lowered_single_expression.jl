# General case: do nothing (identity)
substitute(x, code) = x
substitute(x::Core.SSAValue, code) = substitute(code[x.id], code)
substitute(x::Core.ReturnNode, code) = substitute(code[x.val.id], code)
substitute(s::Symbol, code) = s
# Lower in a module of its own, so that the only references to it are the
# globals that lowering assigns unbound names to (Julia 1.12+), which are
# converted back into the names that were written.
const LOWERING_MODULE = Module(:MultiBroadcastFusionLowering)
substitute(x::GlobalRef, code) = x.mod === LOWERING_MODULE ? x.name : x
# Expression: recursively substitute for Expr
substitute(e::Expr, code) =
    Expr(substitute(e.head, code), substitute.(e.args, Ref(code))...)

code_info(expr) = Base.Meta.lower(LOWERING_MODULE, expr).args[1]
function code_lowered_single_expression(expr)
    code = code_info(expr).code # vector
    s = string(substitute(code[end], code))
    return Base.Meta.parse(s)
end
