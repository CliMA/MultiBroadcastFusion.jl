module MultiBroadcastFusionCUDAExt

import CUDA, Adapt
import MultiBroadcastFusion as MBF
import MultiBroadcastFusion: fused_copyto!

MBF.device(x::CUDA.CuArray) = MBF.MBF_CUDA()

function fused_copyto!(fmb::MBF.FusedMultiBroadcast, ::MBF.MBF_CUDA)
    (; pairs) = fmb
    dest = first(pairs).first
    nitems = length(parent(dest))
    max_threads = 256 # can be higher if conditions permit
    nthreads = min(max_threads, nitems)
    nblocks = cld(nitems, nthreads)
    CUDA.@cuda threads = (nthreads) blocks = (nblocks) fused_copyto_kernel!(fmb)
    return nothing
end
function fused_copyto_kernel!(fmb::MBF.FusedMultiBroadcast)
    (; pairs) = fmb
    dest = first(pairs).first
    nitems = length(dest)
    idx = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if idx ≤ nitems
        MBF.rcopyto_at!(pairs, idx)
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
