using Lux, GNNLux, Graphs, Random, ComponentArrays, Zygote

# Setup
rng = Random.default_rng()
g = erdos_renyi(10, 0.5)
x = randn(rng, 5, 10) # 5 features, 10 nodes

# Define Model with Dropout
gnn = GNNLux.GNNChain(
    GNNLux.GraphConv(5 => 32, tanh; aggr=mean),
    Lux.Dropout(0.2), # The Suspect
    GNNLux.GraphConv(32 => 1; aggr=mean)
)

ps, st = Lux.setup(rng, gnn)
st = Lux.trainmode(st) # Force dropout active

println("1. Testing Forward Pass...")
try
    y, st_new = gnn(g, x, ps, st)
    println("Forward Pass Success! Output shape: ", size(y))
catch e
    println("Forward Pass Failed!")
    println(e)
end

println("\n2. Testing Gradient (Zygote)...")
try
    loss(p) = sum(first(gnn(g, x, p, st)))
    grads = Zygote.gradient(loss, ps)
    println("Gradient Pass Success!")
catch e
    println("Gradient Pass Failed!")
    println(e)
end
