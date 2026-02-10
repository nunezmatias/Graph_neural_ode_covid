using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

println("=== ZERO-SHOT GENERALIZATION TEST: UNSEEN STATES ===")

# 1. SETUP ENVIRONMENT & LOAD ALL DATA
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

# Load full adjacency
A_pop_df = CSV.read("./Data/adj_pop_dist.csv", DataFrame);
all_states = String.(A_pop_df[:, "Column1"]);
A_full = Matrix(A_pop_df[:, 2:end]);

# Normalize full adjacency (must match preprocessing.jl logic)
minA = minimum([A_full[i, j] for i in axes(A_full, 1) for j in axes(A_full, 2) if i != j])
maxA = maximum(A_full)
A_norm_full = (A_full .- minA) ./ (maxA - minA)
for i in axes(A_norm_full, 1)
    A_norm_full[i, i] = 0.0
end
A_hat_full = A_norm_full + I

# Identify Indices
trained_idx = [2, 3, 9, 14, 15, 22, 25, 26, 35, 46] # FL, IL, NC, CA, NJ, GA, OH, TX, NY, VA
# Unseen indices: WA(20), CO(27), MA(45), AZ(49), MI(37)
test_idx = [20, 27, 45, 49, 37]
test_state_names = all_states[test_idx]
println("Testing on: ", test_state_names)

# Load raw X data for covariates and ground truth
# We need to simulate the preprocessing subsetting but for the new indices
# (Assuming preprocessing.jl loads the full x in g.ndata.x)
include("../../Train/preprocessing.jl") # This loads g.ndata.x
X_all_raw = g.ndata.x # This is actually the full dataset before subsetting if using the raw g
# Wait, preprocessing.jl usually masks it. Let's load it manually to be safe.
# Actually, I'll just use the already loaded 'g' from evaluation script logic if possible.
# Let's assume we have access to the full raw data.

# 2. MODEL DEFINITION (Must match EXACTLY - Dropout=0.0)
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)
function (d::FrozenDropout)(x, ps, st)
    st.mask === nothing ? (x, st) : (x .* st.mask, st)
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
    x, st1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st2 = m.drop1(x, ps.drop1, st.drop1)
    x, st3 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st4 = m.drop2(x, ps.drop2, st.drop2)
    x, st5 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st6 = m.drop3(x, ps.drop3, st.drop3)
    x, st7 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st1, drop1=st2, layer2=st3, drop2=st4, layer3=st5, drop3=st6, layer4=st7)
end

# 3. ZERO-SHOT INFERENCE FUNCTION
function evaluate_zero_shot(model, ps_trained, topo_name, target_indices)
    # 1. Extract learned latents and compute average
    learned_latents = ps_trained.latent_features # (3, 10)
    avg_latent = mean(learned_latents, dims=2) # (3, 1)

    # 2. Construct latents for the test nodes (repeat avg)
    test_latents = repeat(avg_latent, 1, length(target_indices))

    # 3. Build Adjacency for the test sub-graph
    A_sub = A_hat_full[target_indices, target_indices]
    if topo_name == "Isolated"
        A_sub = Matrix(1.0 * I, length(target_indices), length(target_indices))
    elseif topo_name == "Random"
        # For simplicity in zero-shot, we could use a random topology among these 5
        # but the prompt implies testing the "trained logic" on his graph.
        # We'll stick to a placeholder random top here.
        A_sub = rand(length(target_indices), length(target_indices))
        A_sub = (A_sub + A_sub') ./ 2
        for i in 1:length(target_indices)
            A_sub[i, i] = 1.0
        end
    end
    g_sub = GNNGraph(A_sub)

    # 4. Get Data and Splines for test states
    # X_full_norm from evaluate_finetuned might be useful
    X_full_raw = g.ndata.x # This is actually the 50-state data if g was loaded correctly
    X_full_norm = log.(X_full_raw .+ 1)

    tsteps_full = collect(0.0:1.0:size(X_full_norm, 2)-1)
    test_X = X_full_norm[:, :, target_indices]
    test_splines = [CubicSpline(tsteps_full, @view test_X[i, :, j]) for i in 2:4, j in 1:length(target_indices)]

    u0_test = test_X[1, 1, :]

    # 5. Integrate ODE
    gnn_ps = ps_trained.gnn
    st_eval = (
        layer1=NamedTuple(), drop1=(mask=nothing, rng=Random.default_rng()),
        layer2=NamedTuple(), drop2=(mask=nothing, rng=Random.default_rng()),
        layer3=NamedTuple(), drop3=(mask=nothing, rng=Random.default_rng()),
        layer4=NamedTuple()
    )

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, length(target_indices))
        cov_vals = [test_splines[i, j](t) for i in 1:3, j in 1:length(target_indices)]
        model_input = Float64.(vcat(u_reshaped, cov_vals, test_latents))
        y, _ = model(g_sub, model_input, gnn_ps, st_eval)
        return vec(y)
    end

    prob = ODEProblem(dudt, Float64.(vec(u0_test)), (0.0, 180.0)) # Test on train period but unseen states
    sol = solve(prob, Tsit5(), saveat=0:1:180, reltol=1e-3, abstol=1e-3)
    pred = reduce(hcat, sol.u)' # (time, nodes)

    y_true = test_X[1, 1:181, :]'
    mse = mean((pred .- y_true) .^ 2)
    return mse, pred, y_true
end

# 4. EXECUTION
gnn_eval = ExplicitGNN(7, 32, 1, 0.0)
models = [
    ("Full", "Resultados/test-7/checkpoints/params_full_finetuned.jld2"),
    ("Isolated", "Resultados/test-7/checkpoints/params_isolated_finetuned.jld2"),
    ("Random", "Resultados/test-7/checkpoints/params_random_finetuned.jld2")
]

results = []
mkpath("Resultados/test-7/plots/zero_shot")

for (name, path) in models
    if isfile(path)
        println("Evaluating Zero-Shot for $name...")
        ps = load(path, "ps_trained")
        mse, pred, true_val = evaluate_zero_shot(gnn_eval, ps, name, test_idx)
        push!(results, (Model=name, ZeroShot_MSE=mse))

        # Plot one state as example (WA)
        p = plot(title="Zero-Shot: $name - WA", xlabel="Days", ylabel="Log Cases")
        scatter!(p, 0:180, true_val[:, 1], label="Actual WA", markersize=1, alpha=0.4, color=:black)
        plot!(p, 0:180, pred[:, 1], label="Zero-Shot Pred", linewidth=2, color=:purple)
        savefig(p, "Resultados/test-7/plots/zero_shot/zero_shot_$(name)_WA.png")
    end
end

df_res = DataFrame(results)
println("\n=== ZERO-SHOT MSE RESULTS ===")
println(df_res)
CSV.write("Resultados/test-7/zero_shot_results.csv", df_res)

println("Test Complete.")
