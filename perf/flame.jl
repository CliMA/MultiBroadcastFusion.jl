#=
using Revise; include(joinpath("perf", "flame.jl"))
=#

import MultiBroadcastFusion as MBF
include(joinpath(pkgdir(MBF), "test", "execution", "utils.jl"))

# ===========================================

has_cuda = CUDA.has_cuda()
AType = has_cuda ? CUDA.CuArray : Array
# arr_size = (prod((50,5,5,6,50)),)
arr_size = (50, 5, 5, 6, 50)
X = get_arrays(:x, arr_size, AType)
Y = get_arrays(:y, arr_size, AType)

function perf_kernel_fused!(X, Y)
    (; x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) = X
    (; y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) = Y
    @fused_direct begin
        @. y1 = x1 + x2 + x3 + x4
        @. y2 = x2 + x3 + x4 + x5
        @. y3 = x3 + x4 + x5 + x6
        @. y4 = x4 + x5 + x6 + x7
        @. y5 = x5 + x6 + x7 + x8
        @. y6 = x6 + x7 + x8 + x9
        @. y7 = x7 + x8 + x9 + x10
    end
end

import Profile, ProfileCanvas
function do_work!(X, Y, N)
    for i in 1:N
        perf_kernel_fused!(X, Y)
    end
end
do_work!(X, Y, 1) # compile

@info "collect profile"
Profile.clear()
prof = Profile.@profile do_work!(X, Y, 100)
results = Profile.fetch()
Profile.clear()

ProfileCanvas.html_file(joinpath("flame.html"), results)
