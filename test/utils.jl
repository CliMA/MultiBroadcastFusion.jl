import Base.Broadcast: broadcasted, instantiate
import BenchmarkTools
import MultiBroadcastFusion as MBF
using Test
import CUDA

function get_array(AType, s)
    return AType(zeros(s...))
end

function get_arrays(sym, s, AType, n = 10)
    println("array_size = $s, array_type = $AType")
    fn = ntuple(i -> Symbol(sym, i), n)
    return (; zip(fn, ntuple(_ -> get_array(AType, s), n))...)
end

# benchmarking
function benchmark_kernel!(f!, args...)
    println("\n--------------------------- $(nameof(typeof(f!))) ")
    trial = benchmark_kernel!(MBF.device(X.x1), f!, args...)
    show(stdout, MIME("text/plain"), trial)
end
benchmark_kernel!(::MBF.GPU, f!, args...) =
    BenchmarkTools.@benchmark CUDA.@sync $f!($args...);
benchmark_kernel!(::MBF.CPU, f!, args...) =
    BenchmarkTools.@benchmark $f!($args...);

function show_diff(A, B)
    for pn in propertynames(A)
        Ai = getproperty(A, pn)
        Bi = getproperty(B, pn)
        @show Ai, abs.(Ai - Bi)
    end
end

function compare(A, B)
    pass = true
    for pn in propertynames(A)
        pass = pass && all(getproperty(A, pn) .== getproperty(B, pn))
    end
    pass || show_diff(A, B)
    return pass
end
function test_kernel!(; fused!, unfused!, X, Y)
    for x in X
        x .= map(_ -> rand(), x)
    end
    for y in Y
        y .= map(_ -> rand(), y)
    end
    X_fused = deepcopy(X)
    X_unfused = deepcopy(X)
    Y_fused = deepcopy(Y)
    Y_unfused = deepcopy(Y)
    fused!(X_fused, Y_fused)
    unfused!(X_unfused, Y_unfused)
    @testset "Test correctness of $(nameof(typeof(fused!)))" begin
        @test compare(X_fused, X_unfused)
        @test compare(Y_fused, Y_unfused)
    end
end
function test_kernel_args!(; fused!, unfused!, args)
    (; X, Y) = args
    for x in X
        x .= rand(size(x)...)
    end
    for y in Y
        y .= rand(size(y)...)
    end
    X_fused = deepcopy(X)
    X_unfused = deepcopy(X)
    Y_fused = deepcopy(Y)
    Y_unfused = deepcopy(Y)
    args_fused = (; X = X_fused, Y = Y_fused)
    args_unfused = (; X = X_unfused, Y = Y_unfused)
    fused!(args_fused)
    unfused!(args_unfused)
    @testset "Test correctness of $(nameof(typeof(fused!)))" begin
        @test compare(X_fused, X_unfused)
        @test compare(Y_fused, Y_unfused)
    end
end
