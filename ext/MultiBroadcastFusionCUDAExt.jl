module MultiBroadcastFusionCUDAExt

import CUDA, Adapt
import MultiBroadcastFusion as MBF
import MultiBroadcastFusion: fused_copyto!

MBF.device(x::CUDA.CuArray) = MBF.MBF_CUDA()

function fused_copyto!(fmb::MBF.FusedMultiBroadcast, ::MBF.MBF_CUDA)
    (; pairs) = fmb
    dest = first(pairs).first
    destinations = map(p -> p.first, pairs)
    all(a -> axes(a) == axes(dest), destinations) ||
        error("Cannot fuse broadcast expressions with unequal broadcast axes")
    nitems = length(parent(dest))
    CI = CartesianIndices(axes(dest))
    kernel =
        CUDA.@cuda always_inline = true launch = false fused_copyto_kernel!(
            fmb,
            CI,
        )
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(nitems, config.threads)
    blocks = cld(nitems, threads)
    kernel(fmb, CI; threads, blocks)
    return destinations
end
import Base.Broadcast
function fused_copyto_kernel!(fmb::MBF.FusedMultiBroadcast, CI)
    @inbounds begin
        (; pairs) = fmb
        dest = first(pairs).first
        nitems = length(dest)
        idx =
            CUDA.threadIdx().x +
            (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x
        if 1 ≤ idx ≤ nitems
            MBF.rcopyto_at!(pairs, CI[idx])
        end
    end
    return nothing
end

adapt_f(to, f::F) where {F} = Adapt.adapt(to, f)
adapt_f(to, ::Type{F}) where {F} = (x...) -> F(x...)

adapt_src(to, src::AbstractArray) = Adapt.adapt(to, src)

function adapt_src(to, bc::Base.Broadcast.Broadcasted)
    Base.Broadcast.Broadcasted(
        bc.style,
        adapt_f(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )
end

function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    fmbc::MBF.FusedMultiBroadcast,
)
    MBF.FusedMultiBroadcast(map(fmbc.pairs) do pair
        dest = pair.first
        src = pair.second
        Pair(Adapt.adapt(to, dest), adapt_src(to, src))
    end)
end

end
