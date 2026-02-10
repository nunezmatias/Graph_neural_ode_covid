using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;
using NPZ

println("=== GENERATING COMPREHENSIVE ZERO-SHOT PLOTS (FULL HORIZON) ===")

# 1. SETUP & DATA
rng = Random.default_rng()
Random.seed!(rng, 42)

println("Loading raw datasets...")
features_raw = npzread("Data/data_filtered.npz")
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states_adj = String.(df_adj[:, 1])
A_pop_raw = Matrix(df_adj[:, 2:end])

# States subset logic matching train group (to ensure consistency)
trained_states = ["FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA"]
println("Trained states: ", trained_states)

# Normalize Adjacency
minA = minimum([A_pop_raw[i, j] for i in axes(A_pop_raw, 1) for j in axes(A_pop_raw, 2) if i != j])
maxA = maximum(A_pop_raw)
A_norm = (A_pop_raw .- minA) ./ (maxA - minA)
for i in axes(A_norm, 1)
    A_norm[i, i] = 0.0
end
A_hat_full = A_norm + I

# Target states
test_states_names = ["WA", "MI", "MA", "AZ", "CO", "PA", "TN", "MD", "WI", "MN", "SC", "AL", "LA", "KY", "OR"]
test_indices = [findfirst(==(s), all_states_adj) for s in test_states_names]

sample_feat = features_raw[test_states_names[1]]
n_vars, n_time = size(sample_feat)
num_test = length(test_states_names)

X_test = zeros(n_vars, n_time, num_test)
for (i, state) in enumerate(test_states_names)
    X_test[:, :, i] = features_raw[state][:, 1:n_time]
end
X_test_norm = log.(X_test .+ 1)
u0_test = X_test_norm[1, 1, :]
tsteps = collect(0.0:1.0:n_time-1)
test_splines = [CubicSpline(tsteps, @view X_test_norm[v, :, n]) for v in 2:n_vars, n in 1:num_test]

# 2. MODEL DEFINITION
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

# 3. GLOBAL PREDICTIONS
function run_full_generalization(param_path, topo_name)
    ps_trained = load(param_path, "ps_trained")
    latent_dim = size(ps_trained.latent_features, 1)
    gnn = ExplicitGNN(1 + 3 + latent_dim, 32, 1, 0.0)

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
    eig_max = maximum(abs.(eigvals(A_sub)))
    A_sub = A_sub ./ eig_max
    g_sub = GNNGraph(A_sub)

    μ_lat, σ_lat = mean(ps_trained.latent_features), std(ps_trained.latent_features)
    new_latents = (randn(rng, latent_dim, num_test) .* σ_lat) .+ μ_lat
    ps_eval = ComponentArray(gnn=ps_trained.gnn, latent_features=new_latents) |> f64
    st_eval = (layer1=NamedTuple(), drop1=(mask=nothing, rng=rng), layer2=NamedTuple(), drop2=(mask=nothing, rng=rng), layer3=NamedTuple(), drop3=(mask=nothing, rng=rng), layer4=NamedTuple())

    function dudt(u, p_ode, t)
        u_r = reshape(u, 1, num_test)
        cov = [test_splines[i, j](t) for i in 1:3, j in 1:num_test]
        model_input = vcat(u_r, cov, p_ode.latent_features)
        y, _ = gnn(g_sub, model_input, p_ode.gnn, st_eval)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_test), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps_eval, saveat=tsteps, reltol=1e-3, abstol=1e-3)
    return reduce(hcat, sol.u) # (Nodes, Time)
end

function calculate_horizon_mse(pred, target, horizons)
    results = Dict()
    n_nodes = size(pred, 1)
    for h in horizons
        idx = min(h + 1, size(pred, 2))
        mse = mean((pred[:, 1:idx] .- target[1:idx, :]') .^ 2)
        results[h] = mse
    end
    return results
end

# 4. EXECUTION & PLOTTING
horizons_to_test = [20, 60, 80, 100, 200, 400]
cases = [
    ("Full", "Resultados/test-7/checkpoints/params_full_finetuned.jld2", "Full", :red),
    ("Isolated", "Resultados/test-7/checkpoints/params_isolated_finetuned.jld2", "Isolated", :green),
    ("Random", "Resultados/test-7/checkpoints/params_random_finetuned.jld2", "Random", :blue)
]

predictions = Dict()
for (name, path, topo, color) in cases
    println("Predicting $name model...")
    predictions[name] = run_full_generalization(path, topo)
end

mkpath("Resultados/test-7/plots/comparative_zeroshot")
y_true = X_test_norm[1, :, :] # (Time, Nodes)

horizon_metrics = []

for (i, state_name) in enumerate(test_states_names)
    println("Processing $state_name...")
    p = plot(title="Zero-Shot Comparative (400 Days): $state_name", xlabel="Days", ylabel="Log Cases", size=(1000, 600), legend=:topleft, thickness_scaling=1.2)

    # Ground Truth
    scatter!(p, tsteps, y_true[:, i], label="Ground Truth", color=:black, markersize=1, alpha=0.3)

    # Model Predictions
    for (name, path, topo, color) in cases
        pred = predictions[name]
        plot!(p, tsteps, pred[i, :], label="Zero-Shot: $name", color=color, linewidth=2, alpha=0.9)

        # Calculate individual metrics for this state
        for h in horizons_to_test
            idx = min(h + 1, size(pred, 2))
            mse_val = mean((pred[i, 1:idx] .- y_true[1:idx, i]) .^ 2)
            push!(horizon_metrics, (State=state_name, Model=name, Horizon=h, MSE=mse_val))
        end
    end

    # Metadata markers
    vline!(p, [180], label="Training Horizon", color=:black, linestyle=:dash, linewidth=1)
    savefig(p, "Resultados/test-7/plots/comparative_zeroshot/$(state_name)_epoch850_expanded.png")
end

# Generate Summary Table
summary_df = DataFrame(horizon_metrics)

# 1. Multi-Horizon Summary (Keep existing)
wide_summary = unstack(summary_df, [:State, :Model], :Horizon, :MSE)
CSV.write("Resultados/test-7/zeroshot_multi_horizon_mse_15states.csv", wide_summary)

# 2. Per-State 400-Day Table (Requested by User)
# Filter for Horizon = 400
df_400 = filter(row -> row.Horizon == 400, summary_df)
per_state_400 = unstack(df_400, :State, :Model, :MSE)

# Add Average Row
avg_full = mean(skipmissing(per_state_400.Full))
avg_iso = mean(skipmissing(per_state_400.Isolated))
avg_rand = mean(skipmissing(per_state_400.Random))
push!(per_state_400, ["AVERAGE", avg_full, avg_iso, avg_rand])

println("\n=== PER-STATE ZERO-SHOT ERRORS (400 DAYS) ===")
println(per_state_400)
CSV.write("Resultados/test-7/zeroshot_per_state_400d_15states.csv", per_state_400)

println("\nAll comparative plots and tables saved to Resultados/test-7/")
