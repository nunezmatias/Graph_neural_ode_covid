using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using NPZ

# 1. GRAPH & DATA RECONSTRUCTION
# ==============================================================================
rng = Random.default_rng()
Random.seed!(rng, 1234)

# Load Adjacency
A_pop = CSV.read("./Data/adj_pop_dist.csv", DataFrame)
states_idx = [2, 3, 9, 14, 15, 22, 25, 26, 35, 46]
all_states = String.(A_pop[:, "Column1"])
train_states_names = all_states[states_idx]
ny_local_idx = findfirst(==("NY"), train_states_names)

println("Training States: ", train_states_names)
println("NY Index: ", ny_local_idx)

A_pop_mat = Matrix(A_pop[:, 2:end])
minA = minimum([A_pop_mat[i, j] for i in axes(A_pop_mat, 1) for j in axes(A_pop_mat, 2) if i != j])
maxA = maximum(A_pop_mat)
A_norm = (A_pop_mat .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A_hat = A_norm + I
A_train = A_hat[states_idx, states_idx]

# --- DYNAMIC GRAPH SETUP ---
g_normal = GNNGraph(A_train)

# Lockdown Graph: Isolate NY
A_lockdown = copy(A_train)
for i in 1:10
    if i != ny_local_idx
        A_lockdown[ny_local_idx, i] = 0.0
        A_lockdown[i, ny_local_idx] = 0.0
    end
end
g_lockdown = GNNGraph(A_lockdown)

# Load Features
features_raw = npzread("Data/data_filtered.npz")
n_time = 401
n_vars = 4
X_full = zeros(n_vars, n_time, length(train_states_names))

for (i, state) in enumerate(train_states_names)
    raw = features_raw[state]
    X_full[:, :, i] = raw[:, 1:n_time]
end

# Log Normalize 
X_log = log.(X_full .+ 1)

# Print units verification
println("\n=== UNITS VERIFICATION ===")
println("Raw cases (Day 1, NY): ", X_full[1, 1, ny_local_idx])
println("Log cases (Day 1, NY): ", X_log[1, 1, ny_local_idx])
println("Raw cases (Day 200, NY): ", X_full[1, 200, ny_local_idx])
println("Log cases (Day 200, NY): ", X_log[1, 200, ny_local_idx])

tsteps = collect(0.0:1.0:(n_time-1))

# 2. MODEL DEFINITION
# ==============================================================================
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
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end
Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (layer1=Lux.initialparameters(rng, m.layer1), drop1=NamedTuple(), layer2=Lux.initialparameters(rng, m.layer2), drop2=NamedTuple(), layer3=Lux.initialparameters(rng, m.layer3), drop3=NamedTuple(), layer4=Lux.initialparameters(rng, m.layer4))
Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1), layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2), layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3), layer4=Lux.initialstates(rng, m.layer4))
function (m::ExplicitGNN)(g, x, ps, st)
    x, st_1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, _ = m.drop1(x, ps.drop1, st.drop1)
    x, st_2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, _ = m.drop2(x, ps.drop2, st.drop2)
    x, st_3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, _ = m.drop3(x, ps.drop3, st.drop3)
    x, st_4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, st
end

# 3. LOAD PARAMETERS
# ==============================================================================
@load "Params/par_opt_test3.jld2" ps_trained
latents_trained = ps_trained.latent_features
latent_dim = size(latents_trained, 1)

# 4. SCENARIO BUILDER 
# ==============================================================================
function build_splines(X_data, intervention_type::Symbol)
    splines = Matrix{CubicSpline}(undef, 3, 10)

    for n in 1:10
        covs = copy(X_data[2:4, :, n])

        if n == ny_local_idx
            if intervention_type == :lockdown
                start_l = 60
                end_l = 120
                duration_reopen = 30
                end_reopen = end_l + duration_reopen

                reduction = 0.10

                covs[:, start_l:end_l] .*= reduction

                for t in (end_l+1):end_reopen
                    progress = (t - end_l) / duration_reopen
                    factor = reduction + (1.0 - reduction) * progress
                    covs[:, t] .*= factor
                end

            elseif intervention_type == :permanent
                covs[:, 60:end] .*= 0.8
            end
        end

        for v in 1:3
            splines[v, n] = CubicSpline(tsteps, covs[v, :])
        end
    end
    return splines
end

splines_base = build_splines(X_log, :baseline)
splines_lock = build_splines(X_log, :lockdown)
splines_perm = build_splines(X_log, :permanent)

# 5. PREDICTION
# ==============================================================================
nin_tot = 1 + 3 + latent_dim
gnn = ExplicitGNN(nin_tot, 32, 1, 0.0)
ps_gnn, st_gnn = Lux.setup(rng, gnn)

function predict(splines_matrix, use_dynamic_graph::Bool)
    function dudt(u, p, t)
        t_c = clamp(t, 0.0, 400.0)

        current_g = g_normal
        if use_dynamic_graph
            if (t_c >= 60.0) && (t_c <= 120.0)
                current_g = g_lockdown
            end
        end

        cov_val = [splines_matrix[v, n](t_c) for v in 1:3, n in 1:10]
        u_reshaped = reshape(u, 1, 10)
        input_mat = vcat(u_reshaped, cov_val, latents_trained)

        y, _ = gnn(current_g, input_mat, ps_trained.gnn, st_gnn)
        return vec(y)
    end

    u0 = X_log[1, 1, :]
    prob = ODEProblem(dudt, u0, (0.0, 400.0))
    sol = solve(prob, Tsit5(), saveat=tsteps, reltol=1e-4, abstol=1e-4)
    return sol
end

# 6. EXECUTE
# ==============================================================================
println("\nRunning Baseline...")
sol_base = predict(splines_base, false)

println("Running Lockdown...")
sol_lock = predict(splines_lock, true)

println("Running Permanent...")
sol_perm = predict(splines_perm, false)

# 7. MULTI-STATE PLOTS
# ==============================================================================
# Find NY's neighbors (states with non-zero edges)
neighbors_idx = []
for i in 1:10
    if i != ny_local_idx && A_train[ny_local_idx, i] > 0.01
        push!(neighbors_idx, i)
    end
end

println("\nNY Neighbors: ", [train_states_names[i] for i in neighbors_idx])

# Create subplot grid
states_to_plot = [ny_local_idx, neighbors_idx[1:min(3, length(neighbors_idx))]...]
n_plots = length(states_to_plot)

p_multi = plot(layout=(2, 2), size=(1400, 1000), dpi=150)

for (plot_idx, state_idx) in enumerate(states_to_plot)
    state_name = train_states_names[state_idx]

    # Extract predictions
    real_data = X_log[1, :, state_idx]
    pred_base = [u[state_idx] for u in sol_base.u]
    pred_lock = [u[state_idx] for u in sol_lock.u]

    # Plot
    scatter!(p_multi[plot_idx], tsteps, real_data,
        label="Real", color=:gray, alpha=0.4, markersize=2, markerstrokewidth=0)

    plot!(p_multi[plot_idx], tsteps, pred_base,
        label="Baseline", lw=2, color=:blue)

    plot!(p_multi[plot_idx], tsteps, pred_lock,
        label="NY Lockdown", lw=2, color=:green, linestyle=:dash)

    # Lockdown regions
    vspan!(p_multi[plot_idx], [60, 120], color=:green, alpha=0.1, label="")
    vspan!(p_multi[plot_idx], [120, 150], color=:yellow, alpha=0.1, label="")

    plot!(p_multi[plot_idx],
        title=state_name,
        xlabel="Days",
        ylabel="Log Cases",
        legend=:topleft)
end

savefig(p_multi, "Resultados/test-6/plots/multi_state_comparison.png")
println("\nMulti-state plot saved.")

# 8. NY DETAILED PLOT
# ==============================================================================
ny_base = [u[ny_local_idx] for u in sol_base.u]
ny_lock = [u[ny_local_idx] for u in sol_lock.u]
ny_perm = [u[ny_local_idx] for u in sol_perm.u]

p = plot(
    title="Counterfactual: NY Lockdown (10% Activity + Border Closure)",
    xlabel="Days",
    ylabel="Log Cases",
    legend=:topleft,
    size=(1000, 600),
    dpi=150
)

scatter!(p, tsteps, X_log[1, :, ny_local_idx],
    label="Real Data", color=:gray, alpha=0.5, markersize=3, markerstrokewidth=0)

plot!(p, tsteps, ny_base, label="Baseline", lw=3, color=:blue, alpha=0.9)
plot!(p, tsteps, ny_lock, label="Lockdown (10% + Isolated)", lw=3, color=:green, linestyle=:solid, alpha=0.9)
plot!(p, tsteps, ny_perm, label="Permanent -20%", lw=3, color=:red, linestyle=:dot, alpha=0.9)

vspan!(p, [60, 120], color=:green, alpha=0.15, label="Lockdown Period")
vspan!(p, [120, 150], color=:yellow, alpha=0.15, label="Reopening")

savefig(p, "Resultados/test-6/plots/NY_counterfactual_final.png")
println("NY plot saved.")
