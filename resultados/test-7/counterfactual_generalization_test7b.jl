using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;
using NPZ

println("=== COUNTERFACTUAL GENERALIZATION TEST (TEST 10b) ===")

# 1. SETUP & DATA (Using NPZ and established zero-shot logic)
rng = Random.default_rng()
Random.seed!(rng, 42)

println("Loading raw datasets from NPZ...")
features_raw = npzread("Data/data_filtered.npz")

# Adjacency
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states_adj = String.(df_adj[:, 1])
A_pop_raw = Matrix(df_adj[:, 2:end])

# Normalize Adjacency (Consistent with preprocessing.jl)
minA = minimum([A_pop_raw[i, j] for i in axes(A_pop_raw, 1) for j in axes(A_pop_raw, 2) if i != j])
maxA = maximum(A_pop_raw)
A_norm = (A_pop_raw .- minA) ./ (maxA - minA)
for i in axes(A_norm, 1)
    A_norm[i, i] = 0.0
end
A_hat_full = A_norm + I

# TARGET UNSEEN STATES
test_states_names = ["WA", "MI", "MA", "AZ", "CO"]
println("Testing on: ", test_states_names)
test_indices = [findfirst(==(s), all_states_adj) for s in test_states_names]

# Prep X data for testing
sample_feat = features_raw[test_states_names[1]]
n_vars, n_time = size(sample_feat)
num_test = length(test_states_names)

X_test = zeros(n_vars, n_time, num_test)
for (i, state) in enumerate(test_states_names)
    X_test[:, :, i] = features_raw[state][:, 1:n_time]
end
X_test = log.(X_test .+ 1)
u0_test = X_test[1, 1, :]
tsteps = collect(0.0:1.0:n_time-1)
test_splines = [CubicSpline(tsteps, @view X_test[v, :, n]) for v in 2:n_vars, n in 1:num_test]

# 2. MODEL DEFINITION (Must match EXACTLY - Width 32, Dropout 0.0)
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

# 3. EVALUATION PROTOCOL
function run_generalization(name, param_path, topo_name)
    if !isfile(param_path)
        println("Missing params for $name at $param_path")
        return nothing
    end

    ps_trained = load(param_path, "ps_trained")
    latent_dim = size(ps_trained.latent_features, 1)
    gnn = ExplicitGNN(1 + 3 + latent_dim, 32, 1, 0.0) # Match Test 7b

    # Adjacency for the 5 states
    A_sub = A_hat_full[test_indices, test_indices]
    if topo_name == "Isolated"
        A_sub = Matrix(1.0 * I, num_test, num_test)
    elseif topo_name == "Random"
        Random.seed!(42)
        A_sub = rand(num_test, num_test)
        A_sub = (A_sub + A_sub') / 2
        for i in 1:num_test
            A_sub[i, i] = 1.0
        end
    end

    # Normalization check
    eig_max = maximum(abs.(eigvals(A_sub)))
    A_sub = A_sub ./ eig_max
    g_sub = GNNGraph(A_sub)

    # Latent Transfer: Initialize with training stats
    μ_lat, σ_lat = mean(ps_trained.latent_features), std(ps_trained.latent_features)
    new_latents = (randn(rng, latent_dim, num_test) .* σ_lat) .+ μ_lat
    ps_eval = ComponentArray(gnn=ps_trained.gnn, latent_features=new_latents) |> f64

    # State initialization
    st_eval = (
        layer1=NamedTuple(), drop1=(mask=nothing, rng=rng),
        layer2=NamedTuple(), drop2=(mask=nothing, rng=rng),
        layer3=NamedTuple(), drop3=(mask=nothing, rng=rng),
        layer4=NamedTuple()
    )

    function dudt(u, p_ode, t)
        u_r = reshape(u, 1, num_test)
        cov = [test_splines[i, j](t) for i in 1:3, j in 1:num_test]
        model_input = vcat(u_r, cov, p_ode.latent_features)
        y, _ = gnn(g_sub, model_input, p_ode.gnn, st_eval)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_test), (0.0, 180.0))
    sol = solve(prob, Tsit5(), p=ps_eval, saveat=0:1:180, reltol=1e-3, abstol=1e-3)
    pred_data = reduce(hcat, sol.u) # (Nodes, Time)
    y_true_data = X_test[1, 1:181, :] # (Time, Nodes)

    # Align both for MSE: y_true_data is (Time, Nodes), so needs transpose
    mse = mean((pred_data .- y_true_data') .^ 2)
    return mse, pred_data, y_true_data
end

# 4. EXECUTION
cases = [
    ("Full", "Resultados/test-7/checkpoints/params_full_finetuned.jld2", "Full"),
    ("Isolated", "Resultados/test-7/checkpoints/params_isolated_finetuned.jld2", "Isolated"),
    ("Random", "Resultados/test-7/checkpoints/params_random_finetuned.jld2", "Random")
]

final_results = []
mkpath("Resultados/test-7/plots/generalization")

for (name, path, topo) in cases
    println("\nRunning Generalization for $name...")
    res = run_generalization(name, path, topo)
    if res !== nothing
        mse, pred, y_true = res
        push!(final_results, (Model=name, Generalization_MSE=mse))

        # Plot example (WA) - WA is index 1 because test_idx=[20,...]
        p = plot(title="Generalization (Unseen): $name - WA", xlabel="Days", ylabel="Log Cases")
        scatter!(p, 0:180, y_true[1:181, 1], label="Actual WA", markersize=1, alpha=0.4, color=:black)
        plot!(p, 0:180, pred[1, :], label="Zero-Shot Pred", linewidth=2, color=:purple)
        savefig(p, "Resultados/test-7/plots/generalization/WA_$(name).png")
    end
end

if !isempty(final_results)
    df = DataFrame(final_results)
    println("\n=== GENERALIZATION RESULTS (UNSEEN) ===")
    println(df)
    CSV.write("Resultados/test-7/generalization_results_finetuned.csv", df)
end

println("All counterfactual tests complete.")
