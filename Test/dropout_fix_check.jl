using Lux, GNNLux, Graphs, Random, ComponentArrays, Zygote

# FIX: Custom Wrapper to handle (g, x) input for Dropout
struct GraphDropout{L} <: Lux.AbstractExplicitLayer
    layer::L
end

GraphDropout(p::Real) = GraphDropout(Lux.Dropout(p))

function Lux.initialparameters(rng::AbstractRNG, d::GraphDropout)
    return Lux.initialparameters(rng, d.layer)
end

function Lux.initialstates(rng::AbstractRNG, d::GraphDropout)
    return Lux.initialstates(rng, d.layer)
end

# The Magic: Discard 'g', apply dropout to 'x', return 'g' and result
function (d::GraphDropout)(g, x, ps, st)
    y, new_st = d.layer(x, ps, st)
    return y, new_st # GNNChain handles the rest if we return (g, y), wait, GNNChain expects just y?
end

# Let's test standard GNNChain behavior first to see what it expects
# GNNLux source says: layer(g, x) or layer(x).

# Alternative approach: Use a wrapper that adheres to GNNLux interface.
# If GNNChain sees a layer that doesn't take 'g', it might try to broadcast.
# Let's try defining a layer that specifically takes (g, x) and passes x to Dropout.

struct GNN_Dropout <: Lux.AbstractExplicitLayer
    p::Float64
end

Lux.initialparameters(rng::AbstractRNG, ::GNN_Dropout) = NamedTuple()
Lux.initialstates(rng::AbstractRNG, ::GNN_Dropout) = (rng=rng,)

function (d::GNN_Dropout)(g, x, ps, st)
    # Apply dropout to x
    if st.training
        mask = rand(st.rng, eltype(x), size(x)) .> d.p
        y = x .* mask ./ (1 - d.p)
    else
        y = x
    end
    return y, st
end

# Setup
rng = Random.default_rng()
g = erdos_renyi(10, 0.5)
x = randn(rng, 5, 10)

# Define Model with CUSTOM Dropout
gnn = GNNLux.GNNChain(
    GNNLux.GraphConv(5 => 32, tanh; aggr=mean),
    GNN_Dropout(0.2), # The Fix
    GNNLux.GraphConv(32 => 1; aggr=mean)
)

ps, st = Lux.setup(rng, gnn)
st = Lux.trainmode(st)

println("1. Testing Forward Pass (Custom Layer)...")
try
    y, st_new = gnn(g, x, ps, st)
    println("Forward Pass Success! Output shape: ", size(y))
catch e
    println("Forward Pass Failed!")
    println(e)
end

println("\n2. Testing Gradient (Custom Layer)...")
try
    loss(p) = sum(first(gnn(g, x, p, st)))
    grads = Zygote.gradient(loss, ps)
    println("Gradient Pass Success!")
catch e
    println("Gradient Pass Failed!")
    println(e)
end
