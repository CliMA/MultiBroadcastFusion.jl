#=
using Revise; include(joinpath("test", "runtests.jl"))
=#
using Test
using SafeTestsets

@safetestset "code_lowered_single_expression" begin @time include("code_lowered_single_expression.jl") end
@safetestset "materialize_args" begin @time include("materialize_args.jl") end
@safetestset "fused_pairs" begin @time include("fused_pairs.jl") end
@safetestset "Multi-Broadcast example" begin @time include("multi_bc_example.jl") end
@safetestset "Multi-Broadcast complex example" begin @time include("complex_example.jl") end
