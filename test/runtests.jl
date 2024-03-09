#=
using Revise; include(joinpath("test", "runtests.jl"))
=#
using Test
using SafeTestsets

#! format: off
@safetestset "expr_code_lowered_single_expression" begin; @time include("expr_code_lowered_single_expression.jl"); end
@safetestset "expr_materialize_args" begin; @time include("expr_materialize_args.jl"); end
@safetestset "expr_fused_pairs" begin; @time include("expr_fused_pairs.jl"); end
@safetestset "expr_errors_and_edge_cases" begin; @time include("expr_errors_and_edge_cases.jl"); end

@safetestset "fused_shared_reads" begin; @time include("bm_fused_shared_reads.jl"); end
@safetestset "fused_shared_reads_writes" begin; @time include("bm_fused_shared_reads_writes.jl"); end
@safetestset "bm_fused_reads_vs_hard_coded" begin; @time include("bm_fused_reads_vs_hard_coded.jl"); end
#! format: on
