using Lux, GNNLux, Graphs, Random, ComponentArrays, Zygote, Statistics

# TRADITIONAL APPROACH: Explicit Layer Definition
# Instead of relying on GNNChain logic, we define the sequence manually.
# This gives us full control over what gets passed to Dropout.

struct ExplicitGNN{L1,D1,L2,D2,L3,D3,L4} <: Lux.AbstractLuxLayer
    layer1::L1
    drop1::D1
    layer2::L2
    drop2::D2
    layer3::L3
    drop3::D3
    layer4::L4
end

function ExplicitGNN(nin, nhidden, nout, drop_p)
    return ExplicitGNN(
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean),
        Lux.Dropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        Lux.Dropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        Lux.Dropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end

# Boilerplate for Params/State (Lux macro usually handles this, but doing manual for safety)
function Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialparameters(rng, m.layer1),
        drop1=Lux.initialparameters(rng, m.drop1),
        layer2=Lux.initialparameters(rng, m.layer2),
        drop2=Lux.initialparameters(rng, m.drop2),
        layer3=Lux.initialparameters(rng, m.layer3),
        drop3=Lux.initialparameters(rng, m.drop3),
        layer4=Lux.initialparameters(rng, m.layer4)
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialstates(rng, m.layer1),
        drop1=Lux.initialstates(rng, m.drop1),
        layer2=Lux.initialstates(rng, m.layer2),
        drop2=Lux.initialstates(rng, m.drop2),
        layer3=Lux.initialstates(rng, m.layer3),
        drop3=Lux.initialstates(rng, m.drop3),
        layer4=Lux.initialstates(rng, m.layer4)
    )
end

# THE CORE FIX: We manually control the flow
function (m::ExplicitGNN)(g, x, ps, st)
    # Layer 1 (Conv) needs g
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)

    # Dropout 1 needs ONLY x
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)

    # Layer 2 (Conv) needs g
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)

    # Dropout 2 needs ONLY x
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)

    # Layer 3 (Conv) needs g
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)

    # Dropout 3 needs ONLY x
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)

    # Layer 4 (Conv) needs g
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)

    # Reconstruct state
    st_new = (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
    return x, st_new
end

# Setup
rng = Random.default_rng()
g = erdos_renyi(10, 0.5)
x = randn(rng, 5, 10)

# Instantiate
model = ExplicitGNN(5, 32, 1, 0.2)
ps, st = Lux.setup(rng, model)
st = Lux.trainmode(st)

println("1. Testing Forward Pass (Explicit Model)...")
try
    y, st_new = model(g, x, ps, st)
    println("Forward Pass Success! Output shape: ", size(y))
catch e
    println("Forward Pass Failed!")
    println(e)
end

println("\n2. Testing Gradient (Explicit Model)...")
try
    loss(p) = sum(first(model(g, x, p, st)))
    grads = Zygote.gradient(loss, ps)
    println("Gradient Pass Success!")
catch e
    println("Gradient Pass Failed!")
    println(e)
end
