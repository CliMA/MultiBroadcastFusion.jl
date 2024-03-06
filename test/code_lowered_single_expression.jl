#=
using Revise; include(joinpath("test", "code_lowered_single_expression.jl"))
=#
using Test
import MultiBroadcastFusion as MBF
function get_array(AType, s)
    return AType(zeros(s...))
end

function get_arrays(sym, s, AType, n=10)
    fn = ntuple(i->Symbol(sym, i), n)
    return (;zip(fn,ntuple(_->get_array(AType, s), n))...)
end
AType = Array;
@show AType
arr_size = (prod((50,5,5,6,50)),)
X = get_arrays(:x, arr_size, AType)
Y = get_arrays(:y, arr_size, AType)
@testset "code_lowered_single_expression" begin
  (;x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = X;
  (;y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) = Y;
  expr_in = :(@. y1 = x1+x2+x3+x4)
  expr_out = :(Base.materialize!(y1, Base.broadcasted(+, x1, x2, x3, x4)))
  @test MBF.code_lowered_single_expression(expr_in) == expr_out
end
