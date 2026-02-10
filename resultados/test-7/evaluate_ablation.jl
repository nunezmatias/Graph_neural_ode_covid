using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

# Setup environment
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

# 1. LOAD DATA & PREPROCESSING
if !isdefined(Main, :covariate_splines)
    include("../../Train/preprocessing.jl")
end

# 2. DEFINE MODEL (Must match training architecture)
latent_dim = 3
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)
function (d::FrozenDropout)(x, ps, st)
    return st.mask === nothing ? (x, st) : (x .* st.mask, st)
end

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
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end

function (m::ExplicitGNN)(g, x, ps, st)
    x, _ = m.layer1(g, x, ps.layer1, st.layer1)
    x, _ = m.drop1(x, ps.drop1, st.drop1)
    x, _ = m.layer2(g, x, ps.layer2, st.layer2)
    x, _ = m.drop2(x, ps.drop2, st.drop2)
    x, _ = m.layer3(g, x, ps.layer3, st.layer3)
    x, _ = m.drop3(x, ps.drop3, st.drop3)
    x, _ = m.layer4(g, x, ps.layer4, st.layer4)
    return x, st
end

# 3. PREDICTION FUNCTION
function get_prediction(model, ps_trained, g_input, tsteps, splines)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        t_clamped = clamp(t, tsteps[1], tsteps[end])
        cov_matrix = [s(t_clamped) for s in splines]
        model_input = vcat(u_reshaped, cov_matrix, p_ode.latent_features)
        y, _ = model(g_input, model_input, p_ode.gnn, Lux.initialstates(Random.default_rng(), model))
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(X_norm[1, 1, :]), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps_trained, saveat=tsteps)
    pred_matrix = reduce(hcat, sol.u)
    return reshape(pred_matrix, 1, n_nodes, length(tsteps))
end

# 4. TOPOLOGY HELPERS
function create_isolated_graph(g_orig)
    A_iso = Matrix(I, g_orig.num_nodes, g_orig.num_nodes) |> f64
    return GNNGraph(A_iso)
end

function create_random_graph(g_orig)
    # This is a bit complex to replicate perfectly, so we load if needed or approximate.
    # For now, we assume the trained model's graph is consistent with its params.
    # We will try to load the JLD2 which should contain the graph if we are lucky,
    # but the scripts only saved ps_trained. 
    # Let's rebuild a random graph with the same seed if possible.
    rng_seed = Random.seed!(42)
    A_orig = Matrix(adjacency_matrix(g_orig))
    n_nodes = g_orig.num_nodes
    n_edges_target = sum(A_orig .> 0.01) - n_nodes
    A_rand = zeros(Float64, n_nodes, n_nodes)
    ecount = 0
    while ecount < n_edges_target
        i, j = rand(1:n_nodes), rand(1:n_nodes)
        if i != j && A_rand[i, j] == 0.0
            w = rand()
            A_rand[i, j] = w
            A_rand[j, i] = w
            ecount += 2
        end
    end
    for i in 1:n_nodes
        A_rand[i, i] = 1.0
    end
    A_rand ./= maximum(A_rand)
    return GNNGraph(A_rand)
end

# 5. EXECUTION
println("=== LOADING PARAMETERS ===")
gnn = ExplicitGNN(nin_tot, 32, nout, 0.05)

paths = [
    ("Full", "Resultados/test-7/checkpoints/params_full.jld2", g),
    ("Isolated", "Resultados/test-7/checkpoints/params_isolated.jld2", create_isolated_graph(g)),
    ("Random", "Resultados/test-7/checkpoints/params_random.jld2", create_random_graph(g))
]

results = []

for (name, path, g_topo) in paths
    if isfile(path)
        println("Loading $name...")
        ps_load = load(path, "ps_trained")
        pred = get_prediction(gnn, ps_load, g_topo, tsteps, covariate_splines)
        push!(results, (name, pred))
    else
        println("Skipping $name: File not found at $path")
    end
end

# 6. PLOTTING COMPARISON
if !isempty(results)
    mkpath("Resultados/test-7/plots/comparison")
    for node_idx in 1:length(states)
        state_name = states[node_idx]
        plt = plot(title="Graph Ablation Study: $state_name", xlabel="Days", ylabel="Log Cases", legend=:outerright)

        # Real Data
        scatter!(plt, tsteps, X_norm[1, :, node_idx], label="Real Data", color=:black, markersize=2, alpha=0.3)

        # Predictions
        colors = [:blue, :green, :red]
        for (i, (name, pred)) in enumerate(results)
            plot!(plt, tsteps, pred[1, node_idx, :], label=name, color=colors[i], linewidth=2)
        end

        savefig(plt, "Resultados/test-7/plots/comparison/ablation_$state_name.png")
    end
    println("Done! Plots saved in Resultados/test-7/plots/comparison/")
else
    println("Error: No prediction data to plot.")
end
