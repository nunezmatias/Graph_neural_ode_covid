# Evaluate Unseen States
using JLD2, ComponentArrays, Lux
using CSV, DataFrames, Statistics, Plots, Measures
using GNNLux, Graphs, NPZ
using DifferentialEquations, SciMLSensitivity, Random, LinearAlgebra, CubicSplines

# 1. Define Splits
trained_states = [
    "FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA",
    "WA", "MI", "MA", "AZ", "CO", "MD", "WI", "MN", "SC", "KY",
    "CT", "MO", "IN", "NM", "NV"
]

# Studied "Bad" States (To highlight)
studied_states = ["TN", "LA", "AL", "OR", "PA"]

# Get All States
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states = names(df_adj)[2:end]
unseen_states = setdiff(all_states, trained_states)

println("Unseen States ($(length(unseen_states))): ", unseen_states)

# Load Data for Unseen
features_raw = NPZ.npzread("Data/data_filtered.npz")
n_times = 401
n_nodes = length(unseen_states)
X_tensor = zeros(Float32, 4, n_times, n_nodes)

valid_unseen = []
valid_idxs = []
for (i, state) in enumerate(unseen_states)
    if haskey(features_raw, state)
        # Find index in output tensor (which we build sequentially)
        idx_out = length(valid_unseen) + 1
        X_tensor[:, :, idx_out] = features_raw[state]
        push!(valid_unseen, state)
        push!(valid_idxs, i)
    else
        println("Warning: Data missing for $state")
    end
end
unseen_states = valid_unseen
n_nodes = length(unseen_states)
X_tensor = X_tensor[:, :, 1:n_nodes]
X_norm = log.(X_tensor .+ 1.0)

# Splines
tsteps = Float32.(collect(0:n_times-1))
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:4, n in 1:n_nodes]

# Adjacency for Unseen (Subgraph)
# We map unseen states to their indices in original CSV
csv_cols = names(df_adj)[2:end]
csv_idxs = [findfirst(==(s), csv_cols) for s in unseen_states]
adj = Matrix(df_adj[:, 2:end])[csv_idxs, csv_idxs]
# Normalize
adj = (adj ./ maximum(adj))
for i in 1:n_nodes
    adj[i, i] = 0.0
end
adj = adj + I
g = GNNGraph(adj)

# 2. Model (Same Definition)
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
# (Boilerplate Omitted for Brevity - Assume Wrapper or Copy logic)
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

# Load Trained Parameters
checkpoint_path = "Resultados/test-8/checkpoints/params_test8_25s.jld2"
@load checkpoint_path ps_trained

# Create New Parameters for Unseen
rng = Random.default_rng()
new_latents = randn(rng, Float32, 3, n_nodes) * 0.1 # Small random init
ps_unseen = ComponentArray(gnn=ps_trained.gnn, latent_features=new_latents) |> f64

gnn = ExplicitGNN(nin_tot, 64, 1, 0.0)
_, st_gnn = Lux.setup(rng, gnn)


# 3. Prediction
function predict_full(model, ps, st, u0, tsteps, splines)
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

println("Predicting Unseen...")
u0 = X_norm[1, 1, :]
pred = predict_full(gnn, ps_unseen, st_gnn, u0, tsteps, covariate_splines)

# 4. Plots & Tables
plots_array = []
mse_list = []
train_idx = 1:180
test_idx = 181:n_times

mkpath("Resultados/test-8/plots_unseen")

for i in 1:n_nodes
    state_name = unseen_states[i]

    # Check if "Studied"
    is_studied = state_name in studied_states
    title_str = is_studied ? "**$state_name**" : state_name

    data_c = X_norm[1, :, i]
    pred_c = pred[1, :, i]

    mse_tr = mean(abs2, data_c[train_idx] .- pred_c[train_idx])
    mse_te = mean(abs2, data_c[test_idx] .- pred_c[test_idx])

    # Peak Shift
    # Focus on test period 181:end
    d_curve_test = data_c[test_idx]
    p_curve_test = pred_c[test_idx]

    peak_d = argmax(d_curve_test)
    peak_p = argmax(p_curve_test)
    shift = peak_p - peak_d
    abs_shift = abs(shift)

    push!(mse_list, (State=state_name, Train_MSE=mse_tr, Test_MSE=mse_te, Studied=is_studied, Peak_Shift=shift, Abs_Peak_Shift=abs_shift))

    p = plot(tsteps, data_c, label="Data", color=:black, alpha=0.5)
    plot!(p, tsteps, pred_c, label="Pred", color=is_studied ? :blue : :red, linewidth=2)
    vline!(p, [180], color=:gray, linestyle=:dash, label="")
    title!(p, "$title_str (Test: $(round(mse_te, digits=2)), Shift: $shift)")
    push!(plots_array, p)
end

# Master Plot (Grid depends on N)
n_cols = 5
n_rows = ceil(Int, n_nodes / n_cols)
plot(plots_array..., layout=(n_rows, n_cols), size=(1500, 300 * n_rows), margin=2mm)
savefig("Resultados/test-8/plots_unseen/unseen_forecast.png")

# Table
df = DataFrame(mse_list)
sort!(df, :Test_MSE)
CSV.write("Resultados/test-8/unseen_mse.csv", df)

# Peak Shift Plots (Histogram)
p_shift = histogram(df.Peak_Shift, bins=-40:5:10,
    xlabel="Peak Shift (Days) [Pred - Data]", ylabel="Count",
    title="Unseen States: Peak Synchronization", legend=false,
    color=:red, alpha=0.7, size=(800, 600))
vline!(p_shift, [0], color=:black, linewidth=2, linestyle=:dash, label="Perfect Sync")
savefig(p_shift, "Resultados/test-8/plots_unseen/peak_shift_hist_unseen.png")

println(df)
