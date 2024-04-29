#=
julia --project; using Revise; include("demo.jl")
=#

#=
A motivating example of this package is the following:

```julia
x1 = rand(3,3); x2 = rand(3,3)
y1 = rand(3,3); y2 = rand(3,3)

# 2 writes, 2 unique reads, 4 redundant reads
@. y1 = x1 * x2
@. y2 = x1 + x2
```

One way to fuse: change the memory layout + implementation:
```julia
X = map(x->Tuple(rand(2)),zeros(3,3));
Y = map(x->Tuple(rand(2)),zeros(3,3));
foo(x) = (x[1] * x[2], x[1] + x[2])
@. Y = foo(X) # 2 reads, 2 writes
```
But, ideally, we instead have:
```
x1 = rand(3,3); x2 = rand(3,3)
y1 = rand(3,3); y2 = rand(3,3)

# 2 writes, 2 unique reads. The compiler hoists memory reads:
for i in eachindex(x1,x2,x3,x4,y1,y2)
    y1[i] = x1[i] * x2[i]
    y2[i] = x1[i] + x2[i]
end
```
=#

include(joinpath("test", "execution", "utils.jl"))
import MultiBroadcastFusion as MBF

import CUDA
T = CUDA.has_cuda() ? CUDA.CuArray : Array
@show T
arr_size = (50, 5, 5, 6, 50)
X = get_arrays(:x, arr_size, T);
Y = get_arrays(:y, arr_size, T);

function fused!(X, Y)
    (;y1, y2) = Y; (;x1, x2, x3, x4) = X
    MBF.@fused_direct begin
      @. y1 = x1 * x2 + x3 * x4
      @. y2 = x1 * x3 + x2 * x4
    end
end
function unfused!(X, Y)
    (;y1, y2) = Y; (;x1, x2, x3, x4) = X
    @. y1 = x1 * x2 + x3 * x4
    @. y2 = x1 * x3 + x2 * x4
end
# function fused_loop!(X, Y)
#     (;y1, y2) = Y; (;x1, x2, x3, x4) = X
#     @fused_flexible begin
#         for i in 1:10
#             @. y1 += x1 * x2 + x3 * x4
#             @. y2 += x1 * x3 + x2 * x4
#         end
#     end
# end
# function unfused_loop!(X, Y)
#     (;y1, y2) = Y; (;x1, x2, x3, x4) = X
#     for i in 1:10
#         @. y1 += x1 * x2 + x3 * x4
#         @. y2 += x1 * x3 + x2 * x4
#     end
# end

test_kernel!(; fused!, unfused!, X, Y)
# test_kernel!(; fused! = fused_loop!, unfused! = unfused_loop!, X, Y)
benchmark_kernel!(unfused!, X, Y)
benchmark_kernel!(fused!, X, Y)
# benchmark_kernel!(unfused_loop!, X, Y)
# benchmark_kernel!(fused_loop!, X, Y)

