module MultiBroadcastFusion

include(joinpath("collection", "utils.jl"))
include(joinpath("collection", "macros.jl"))
include(joinpath("collection", "code_lowered_single_expression.jl"))
include(joinpath("collection", "fused_pairs.jl"))
include(joinpath("collection", "fused_pairs_flexible.jl"))

include(joinpath("execution", "fused_kernels.jl"))

end # module MultiBroadcastFusion
