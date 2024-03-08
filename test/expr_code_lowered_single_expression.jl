#=
using Revise; include(joinpath("test", "expr_code_lowered_single_expression.jl"))
=#
using Test
import MultiBroadcastFusion as MBF

@testset "code_lowered_single_expression" begin
    expr_in = :(@. y1 = x1 + x2 + x3 + x4)
    expr_out = :(Base.materialize!(y1, Base.broadcasted(+, x1, x2, x3, x4)))
    @test MBF.code_lowered_single_expression(expr_in) == expr_out
end
