# Test 8 Evaluation Script (Final)
using JLD2, ComponentArrays, Lux
using CSV, DataFrames, Statistics, Plots, Measures
using GNNLux, Graphs, NPZ
using DifferentialEquations, SciMLSensitivity, Random, LinearAlgebra, CubicSplines

# -------------------------------------------------------------
# 1. Setup & Data Loading
# -------------------------------------------------------------
println("Loading Evaluation Context...")

target_states = [
    "FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA",
    "WA", "MI", "MA", "AZ", "CO", "MD", "WI", "MN", "SC", "KY",
    "CT", "MO", "IN", "NM", "NV"
]

features_raw = NPZ.npzread("Data/data_filtered.npz")
n_times = 401
n_nodes = 25
X_tensor = zeros(Float32, 4, n_times, n_nodes)
for (i, state) in enumerate(target_states)
    X_tensor[:, :, i] = features_raw[state]
end
X_norm = log.(X_tensor .+ 1.0) # Normalized Data

# Splines
tsteps = Float32.(collect(0:n_times-1))
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:4, n in 1:25]

# Adjacency
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
cols = names(df_adj)[2:end]
idxs = [findfirst(==(s), cols) for s in target_states]
adj = Matrix(df_adj[:, 2:end])[idxs, idxs]

# Normalize
adj = (adj ./ maximum(adj))
for i in 1:25
    adj[i, i] = 0.0
end
adj = adj + I
g = GNNGraph(adj)

println("Data Loaded.")

# -------------------------------------------------------------
# 2. Model Definition (Must match Training exactly)
# -------------------------------------------------------------
latent_dim = 3
nin_tot = 1 + 3 + latent_dim

struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(r::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(r::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=r)
(d::FrozenDropout)(x, ps, st) = st.mask === nothing ? (x, st) : (x .* st.mask, st)

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

Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialparameters(rng, m.layer1), drop1=Lux.initialparameters(rng, m.drop1),
    layer2=Lux.initialparameters(rng, m.layer2), drop2=Lux.initialparameters(rng, m.drop2),
    layer3=Lux.initialparameters(rng, m.layer3), drop3=Lux.initialparameters(rng, m.drop3),
    layer4=Lux.initialparameters(rng, m.layer4)
)
Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1),
    layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2),
    layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3),
    layer4=Lux.initialstates(rng, m.layer4)
)
function (m::ExplicitGNN)(g, x, ps, st)
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
end

println("Model Structure Defined.")

# Initialize Empty
rng = Random.default_rng()
gnn = ExplicitGNN(nin_tot, 64, 1, 0.0)
ps_init, st_gnn = Lux.setup(rng, gnn)

# Load Checkpoint
checkpoint_path = "Resultados/test-8/checkpoints/params_test8_25s.jld2"
if !isfile(checkpoint_path)
    println("ERROR: Checkpoint file $checkpoint_path not found.")
    println("Using dummy weights for testing script structure? NO. Exiting.")
    exit(1)
end

println("Loading Trained Weights...")
@load checkpoint_path ps_trained
println("Weights Loaded.")

# -------------------------------------------------------------
# 3. Prediction
# -------------------------------------------------------------

function predict_full(model, ps, st, u0, tsteps, splines)
    # No dropout sampling needed for prediction (p=0 or mean field)
    # But we need a dummy mask state if using FrozenDropout? 
    # Actually p=0, so mask is 1s.

    # We need to construct the frozen state manually if the predict function expects it?
    # Or just rely on st.mask being nothing -> pass through?
    # Yes, FrozenDropout code handles nothing.

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        col_t = map(s -> s(t), splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, col_t, latents)
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps, saveat=tsteps, reltol=1e-5, abstol=1e-6)

    sol_mat = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

println("Predicting...")
u0 = X_norm[1, 1, :]
pred = predict_full(gnn, ps_trained, st_gnn, u0, tsteps, covariate_splines)

# -------------------------------------------------------------
# 4. Analysis & Plotting
# -------------------------------------------------------------
mkpath("Resultados/test-8/plots")

plots_array = []
mse_list = []

# Define splits
train_idx = 1:180
test_idx = 181:n_times

for i in 1:25
    state_name = target_states[i]

    # Data vs Pred
    data_curve = X_norm[1, :, i]
    pred_curve = pred[1, :, i]

    # Metrics
    mse_total = mean(abs2, data_curve .- pred_curve)
    mse_train = mean(abs2, data_curve[train_idx] .- pred_curve[train_idx])
    mse_test = mean(abs2, data_curve[test_idx] .- pred_curve[test_idx])

    push!(mse_list, (State=state_name, Train_MSE=mse_train, Test_MSE=mse_test, Total_MSE=mse_total))

    # Plot
    p = plot(tsteps, data_curve, label="Data", color=:black, alpha=0.5, linewidth=1.5, legend=false)
    plot!(p, tsteps, pred_curve, label="Pred", color=:red, linewidth=2)
    # Vertical line for Train/Test split
    vline!(p, [180], color=:blue, linestyle=:dash)
    title!(p, "$state_name (Test MSE: $(round(mse_test, digits=3)))")
    push!(plots_array, p)
end

# Master Plot
master_plot = plot(plots_array..., layout=(5, 5), size=(1500, 1200), margin=2mm)
savefig(master_plot, "Resultados/test-8/plots/all_states_forecast_split.png")

# Peak Shift Analysis
peak_shifts = []
scatter_data = []

for row in mse_list
    state = row.State
    idx = findfirst(==(state), target_states)

    # Get curves (181:end)
    # Re-extract because we didn't store full curves in mse_list
    d_curve = X_norm[1, 181:end, idx]
    p_curve = pred[1, 181:end, idx]

    # Find Peaks
    peak_d = argmax(d_curve) # Index in 181:end
    peak_p = argmax(p_curve)

    shift = peak_p - peak_d # Positive = Late, Negative = Early
    distance_days = abs(shift)

    push!(peak_shifts, shift)
    push!(scatter_data, (State=state, Shift=shift, AbsShift=distance_days, TestMSE=row.Test_MSE))
end

# Merge into DataFrame
df_peaks = DataFrame(scatter_data)
df_final = innerjoin(DataFrame(mse_list), df_peaks, on=:State, makeunique=true) # Join or just use df_peaks if it has everything
# Actually mse_list is a Vector of NamedTuples, easier to just build new DF.
df_combined = DataFrame(mse_list)
df_combined.Peak_Shift = df_peaks.Shift
df_combined.Abs_Peak_Shift = df_peaks.AbsShift

sort!(df_combined, :Test_MSE)
CSV.write("Resultados/test-8/detailed_mse_errors.csv", df_combined)

# Plot Peak Shift (Histogram)
p_shift = histogram(df_combined.Peak_Shift, bins=-20:2:10,
    xlabel="Peak Shift (Days) [Pred - Data]", ylabel="Count",
    title="Trained States: Peak Synchronization", legend=false,
    color=:blue, alpha=0.7, size=(800, 600))
vline!(p_shift, [0], color=:black, linewidth=2, linestyle=:dash, label="Perfect Sync")
savefig(p_shift, "Resultados/test-8/plots/peak_shift_hist_trained.png")

println("\n=== DETAILED SUMMARY (Sorted by Test MSE) ===")
println(df_combined)
