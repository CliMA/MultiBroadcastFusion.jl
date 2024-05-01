#=
using Revise; include(joinpath("test", "collection", "runtests.jl"))
using Revise; include(joinpath("test", "execution", "runtests.jl"))
using Revise; include(joinpath("test", "runtests.jl"))
=#

include(joinpath("collection", "runtests.jl"))
include(joinpath("execution", "runtests.jl"))
