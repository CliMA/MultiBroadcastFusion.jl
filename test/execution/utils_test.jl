using Test

function __rprint_diff(
    io::IO,
    x::T,
    y::T;
    pc,
    xname,
    yname,
) where {T <: NamedTuple}
    for pn in propertynames(x)
        pc_full = (pc..., ".", pn)
        xi = getproperty(x, pn)
        yi = getproperty(y, pn)
        __rprint_diff(io, xi, yi; pc = pc_full, xname, yname)
    end
end;

function __rprint_diff(io::IO, xi, yi; pc, xname, yname) # assume we can compute difference here
    if !(xi == yi)
        xs = xname * string(join(pc))
        ys = yname * string(join(pc))
        println(io, "==================== Difference found:")
        println(io, "$xs: ", xi)
        println(io, "$ys: ", yi)
        println(io, "($xs .- $ys): ", (xi .- yi))
    end
    return nothing
end

"""
    rprint_diff(io::IO, ::T, ::T) where {T <: NamedTuple}
    rprint_diff(::T, ::T) where {T <: NamedTuple}

Recursively print differences in given `NamedTuple`.
"""
_rprint_diff(io::IO, x::T, y::T, xname, yname) where {T <: NamedTuple} =
    __rprint_diff(io, x, y; pc = (), xname, yname)
_rprint_diff(x::T, y::T, xname, yname) where {T <: NamedTuple} =
    _rprint_diff(stdout, x, y, xname, yname)

"""
    @rprint_diff(::T, ::T) where {T <: NamedTuple}

Recursively print differences in given `NamedTuple`.
"""
macro rprint_diff(x, y)
    return :(_rprint_diff(
        stdout,
        $(esc(x)),
        $(esc(y)),
        $(string(x)),
        $(string(y)),
    ))
end


# Recursively compare contents of similar types
_rcompare(pass, x::T, y::T) where {T} = pass && (x == y)

function _rcompare(pass, x::T, y::T) where {T <: NamedTuple}
    for pn in propertynames(x)
        pass &= _rcompare(pass, getproperty(x, pn), getproperty(y, pn))
    end
    return pass
end

"""
    rcompare(x::T, y::T) where {T <: NamedTuple}

Recursively compare given types via `==`.
Returns `true` if `x == y` recursively.
"""
rcompare(x::T, y::T) where {T <: NamedTuple} = _rcompare(true, x, y)
rcompare(x, y) = false

function test_compare(x, y)
    if !rcompare(x, y)
        @rprint_diff(x, y)
    else
        @test rcompare(x, y)
    end
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
        test_compare(X_fused, X_unfused)
        test_compare(Y_fused, Y_unfused)
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
        test_compare(X_fused, X_unfused)
        test_compare(Y_fused, Y_unfused)
    end
end
