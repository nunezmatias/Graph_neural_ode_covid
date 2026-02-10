using Lux, GNNLux, Graphs, Random, ComponentArrays, JLD2, Plots, Statistics, DifferentialEquations, SciMLSensitivity, Interpolations, CSV, DataFrames, JSON, CubicSplines

# ==============================================================================
# 0. MODEL DEFINITIONS (Must match training)
# ==============================================================================
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    # [Math] Implements: hat{W}_l = W_l * diag(z_l)
    # The mask 'z_l' is sampled once and passed via 'st' (frozen state).
    if st.mask === nothing
        return x, st
    else
        return x .* st.mask, st
    end
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

function (m::ExplicitGNN)(g, x, ps, st)
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    st_new = (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
    return x, st_new
end

function sample_dropout_masks(model, st, x_shape)
    # [Math] Implements: z_l ~ Bernoulli(1 - p)
    # We sample binary masks for each dropout layer.
    d1 = model.drop1
    mask1 = rand(st.drop1.rng, Float32, (32, x_shape[2])) .> d1.p
    mask1 = mask1 ./ (1 - d1.p)
    st_d1 = (mask=mask1, rng=st.drop1.rng)

    d2 = model.drop2
    mask2 = rand(st.drop2.rng, Float32, (32, x_shape[2])) .> d2.p
    mask2 = mask2 ./ (1 - d2.p)
    st_d2 = (mask=mask2, rng=st.drop2.rng)

    d3 = model.drop3
    mask3 = rand(st.drop3.rng, Float32, (32, x_shape[2])) .> d3.p
    mask3 = mask3 ./ (1 - d3.p)
    st_d3 = (mask=mask3, rng=st.drop3.rng)

    return (
        layer1=st.layer1, drop1=st_d1,
        layer2=st.layer2, drop2=st_d2,
        layer3=st.layer3, drop3=st_d3,
        layer4=st.layer4
    )
end

# ==============================================================================
# 1. LOAD FULL DATA (0 - 400 Days)
# ==============================================================================
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

# Reconstruct Adjacency
A_pop = CSV.read("./Data/adj_pop_dist.csv", DataFrame);
states_idx = [2, 3, 9, 14, 15, 22, 25, 26, 35, 46]
states = String.(A_pop[:, "Column1"][states_idx])
A_pop = Matrix(A_pop[:, 2:end]);
minA = minimum([A_pop[i, j] for i in axes(A_pop, 1), j in axes(A_pop, 2) if i != j])
maxA = maximum(A_pop)
A_norm = (A_pop .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A = A_norm[states_idx, states_idx]

# Build Graph
g = build_GNNGraph(A)

# Process Full Data
X_raw = Float64.(g.ndata.x) # (Vars, Time, Nodes)
X_norm = log.(X_raw .+ 1)   # Normalize

println("Full X_norm size: ", size(X_norm))
# We have 401 time steps
tsteps_full = collect(0.0:1.0:size(X_norm, 2)-1)

# Create FULL Splines (Covering 0 -> 400)
# variables: [2, 3, 4] are covariates (Mobility, etc.)
covariate_splines_full = [CubicSpline(tsteps_full, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:size(X_norm, 3)]

# Initial Condition (Day 0)
u0 = X_norm[1, 1, :]
n_nodes = length(u0)

# ==============================================================================
# 2. LOAD MODEL
# ==============================================================================
rng = Random.default_rng()
latent_dim = 3
nin_tot = 1 + 3 + latent_dim
nout = 1

model = ExplicitGNN(nin_tot, 32, nout, 0.05) # Remember we changed to 0.05
ps_init, st_init = Lux.setup(rng, model)

println("Loading parameters from Params/par_opt_test3.jld2...")
@load "Params/par_opt_test3.jld2" ps_trained

# ==============================================================================
# 3. RUN MONTE CARLO (TRAIN + TEST)
# ==============================================================================
N_SAMPLES = 100
t_end_sim = 400.0
tsteps_sim = collect(0.0:1.0:t_end_sim)

println("Running MC Uncertainty (0 -> $t_end_sim)...")
mc_predictions = zeros(N_SAMPLES, length(tsteps_sim), n_nodes)

# [Math] Monte Carlo (MC) Integration
# We approximate the predictive distribution by averaging T samples:
# E[y] approx 1/T * sum(f(x, W_t))
for i in 1:N_SAMPLES
    if i % 10 == 0
        println("Sample $i / $N_SAMPLES")
    end

    # Sample dropout mask
    st_frozen = sample_dropout_masks(model, st_init, (32, n_nodes))

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)

        # USE FULL SPLINES without clamping to 180
        # This feeds the REAL future mobility data to the model
        cov_matrix = map(s -> s(t), covariate_splines_full)

        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)

        # Forward pass (Frozen)
        y, _ = model(g, model_input, p_ode.gnn, st_frozen)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (0.0, t_end_sim))
    sol = solve(prob, Tsit5(), p=ps_trained, saveat=tsteps_sim, reltol=1e-5, abstol=1e-6)

    if length(sol.t) == length(tsteps_sim)
        pred_matrix = reduce(hcat, sol.u)
        pred_linear = exp.(pred_matrix) .- 1
        pred_linear = max.(0.0, pred_linear) # Force non-negative cases
        mc_predictions[i, :, :] = permutedims(pred_linear, (2, 1))
    else
        println("Integration failed sample $i")
    end
end

# ==============================================================================
# 4. PLOTTING
# ==============================================================================
mkpath("plots/test3_uncertainty_full")

pred_mean = dropdims(mean(mc_predictions, dims=1), dims=1)
# Use quantiles for asymmetric, physically valid intervals
pred_lower_95 = dropdims(mapslices(x -> quantile(x, 0.025), mc_predictions, dims=1), dims=1)
pred_upper_95 = dropdims(mapslices(x -> quantile(x, 0.975), mc_predictions, dims=1), dims=1)

pred_lower_50 = dropdims(mapslices(x -> quantile(x, 0.25), mc_predictions, dims=1), dims=1)
pred_upper_50 = dropdims(mapslices(x -> quantile(x, 0.75), mc_predictions, dims=1), dims=1)

# Plot for each state
for (node_idx, state_name) in enumerate(states)

    # Band Plot
    p = plot(title="$state_name: Forecast (Train + Test)", xlabel="Days", ylabel="Cases", size=(800, 500))

    # 1. Uncertainty Band (95% CI - Light Blue)
    rib_low_95 = pred_mean[:, node_idx] .- pred_lower_95[:, node_idx]
    rib_high_95 = pred_upper_95[:, node_idx] .- pred_mean[:, node_idx]

    plot!(p, tsteps_sim, pred_mean[:, node_idx],
        ribbon=(rib_low_95, rib_high_95),
        fillalpha=0.2, fillcolor=:blue, linecolor=:blue, linewidth=0, label="95% CI")

    # 2. Uncertainty Band (50% CI - Darker Blue)
    rib_low_50 = pred_mean[:, node_idx] .- pred_lower_50[:, node_idx]
    rib_high_50 = pred_upper_50[:, node_idx] .- pred_mean[:, node_idx]

    plot!(p, tsteps_sim, pred_mean[:, node_idx],
        ribbon=(rib_low_50, rib_high_50),
        fillalpha=0.4, fillcolor=:blue, linecolor=:blue, linewidth=2, label="Prediction (MEAN + 50% CI)")

    # 3. Real Data (Full)
    real_data = exp.(X_norm[1, :, node_idx]) .- 1
    scatter!(p, 1:length(real_data), real_data, color=:black, markersize=2, label="Real Data", alpha=0.6)

    # 4. Vertical Line for Train/Test Split
    plot!(p, [180, 180], [0, maximum(real_data)], color=:red, linestyle=:dash, label="Train/Test Split (Day 180)")

    savefig(p, "plots/test3_uncertainty_full/$(state_name)_forecast_dual.png")
end
println("Min Lower Bound value across all states: ", minimum(pred_lower))
println("Done. Plots in plots/test3_uncertainty_full/")
