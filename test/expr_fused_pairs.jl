#=
using Revise; include(joinpath("test", "expr_fused_pairs.jl"))
=#
using Test
import MultiBroadcastFusion as MBF
@testset "fused_pairs" begin
  expr_in = quote
    @. y1 = x1+x2+x3+x4
    @. y2 = x2+x3+x4+x5
  end

  expr_out = :((
      Pair(y1, Base.broadcasted(+, x1, x2, x3, x4)),
      Pair(y2, Base.broadcasted(+, x2, x3, x4, x5)),
    ))
  @test MBF.fused_pairs(expr_in) == expr_out
end

@testset "fused_multibroadcast" begin
  expr_in = quote
    @. y1 = x1+x2+x3+x4
    @. y2 = x2+x3+x4+x5
  end

  expr_out = :(MultiBroadcastFusion.FusedMultiBroadcast((
      Pair(y1, Base.broadcasted(+, x1, x2, x3, x4)),
      Pair(y2, Base.broadcasted(+, x2, x3, x4, x5)),
    )))
  @test MBF.fused_multibroadcast(MBF.FusedMultiBroadcast, expr_in) == expr_out
end
