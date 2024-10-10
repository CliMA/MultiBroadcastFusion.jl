module MultiBroadcastFusionCUDAExt

import CUDA, Adapt
import MultiBroadcastFusion as MBF

MBF.device(x::CUDA.CuArray) = MBF.MBF_CUDA()

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
