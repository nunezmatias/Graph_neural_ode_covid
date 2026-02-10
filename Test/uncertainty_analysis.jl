using Lux, GNNLux, Graphs, Random, ComponentArrays, JLD2, Plots, Statistics, DifferentialEquations, SciMLSensitivity, Interpolations, CSV, DataFrames

# 1. DEFINICIONES DE MODELO (Deben coincidir EXACTAMENTE con el entrenamiento)
# ==============================================================================

# Frozen Dropout Layer
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    if st.mask === nothing
        # En inferencia normal (sin máscaras pre-definidas), actuamos según modo:
        # Si queremos MC Dropout, DEBEMOS proveer máscaras en 'st'.
        # Si no hay máscaras, pasamos directo (Scaling rule se aplica si fuera dropout normal, 
        # pero aquí asumimos que en test "promedio" usamos la red completa o muestreamos).
        # Para MC dropout, SIEMPRE usaremos masks generadas externamente.
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

# 2. CARGA DE DATOS Y MODELO
# ==============================================================================
include("../Train/preprocessing.jl") # Load data
println("Data loaded. X_norm size: ", size(X_norm))

rng = Random.default_rng()
latent_dim = 3
nin_tot = 1 + 3 + latent_dim
nout = 1
n_nodes = g.num_nodes

# Instantiate Model
model = ExplicitGNN(nin_tot, 32, nout, 0.2)
ps_init, st_init = Lux.setup(rng, model)

# Load Learned Parameters
println("Loading parameters from Params/par_opt_test3.jld2...")
@load "Params/par_opt_test3.jld2" ps_trained

# 3. MONTE CARLO SAMPLING
# ==============================================================================
N_SAMPLES = 100
t_start = 1
t_end = 400 # Extrapolate into future
tsteps_ext = range(t_start, t_end, length=400)

println("Running MC Uncertainty Analysis (N=$N_SAMPLES)...")

# Output tensor: (samples, time, nodes)
mc_predictions = zeros(N_SAMPLES, length(tsteps_ext), n_nodes)

for i in 1:N_SAMPLES
    if i % 10 == 0
        println("Sample $i / $N_SAMPLES")
    end

    # 1. Sample NEW masks for this trajectory
    st_frozen = sample_dropout_masks(model, st_init, (32, n_nodes))

    # 2. Define ODE for this sample
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        t_clamped = clamp(t, tsteps[1], tsteps[end]) # Use training range splines
        cov_matrix = [s(t_clamped) for s in covariate_splines]
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)

        # Forward pass with FROZEN masks
        y, _ = model(g, model_input, p_ode.gnn, st_frozen)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps_ext[1], tsteps_ext[end]))
    sol = solve(prob, Tsit5(), p=ps_trained, saveat=tsteps_ext, reltol=1e-5, abstol=1e-6)

    # Store result (Linear Scale: exp(x) - 1)
    if length(sol.t) == length(tsteps_ext)
        pred_matrix = reduce(hcat, sol.u) # (n_nodes, time)
        pred_linear = exp.(pred_matrix) .- 1
        mc_predictions[i, :, :] = permutedims(pred_linear, (2, 1))
    else
        println("Warning: Solver failed for sample $i")
    end
end

# 4. ESTADITICAS Y PLOTTING
# ==============================================================================
mkpath("plots/test3_uncertainty")

# Calculate Mean and Std
pred_mean = dropdims(mean(mc_predictions, dims=1), dims=1) # (time, nodes)
pred_std = dropdims(std(mc_predictions, dims=1), dims=1)   # (time, nodes)

# 95% Confidence Interval
ci_upper = pred_mean .+ 1.96 .* pred_std
ci_lower = pred_mean .- 1.96 .* pred_std

# Plotting specific states
states_to_plot = ["OH", "NY", "TX", "CA"] # Main examples

for (node_idx, state_name) in enumerate(states)
    if state_name in states_to_plot || true # Plot all for report

        # A. SPAGHETTI PLOT
        p1 = plot(title="$state_name: Multiverse (100 Samples)", xlabel="Days", ylabel="Cases", legend=false)
        for i in 1:min(50, N_SAMPLES) # Plot first 50 lines
            plot!(p1, tsteps_ext, mc_predictions[i, :, node_idx],
                alpha=0.2, color=:blue, linewidth=0.5)
        end
        # Add Real Data (scatter)
        real_data = exp.(X_norm[1, :, node_idx]) .- 1
        scatter!(p1, 1:length(real_data), real_data, color=:black, markersize=2, label="Real")
        savefig(p1, "plots/test3_uncertainty/$(state_name)_spaghetti.png")

        # B. BAND PLOT (Report Quality)
        p2 = plot(title="$state_name: Uncertainty Forecast", xlabel="Days", ylabel="Cases")

        # Ribbon (Confidence Interval)
        plot!(p2, tsteps_ext, pred_mean[:, node_idx],
            ribbon=(1.96 .* pred_std[:, node_idx], 1.96 .* pred_std[:, node_idx]),
            fillalpha=0.3, fillcolor=:blue,
            linewidth=2, color=:blue, label="Mean Prediction (95% CI)")

        # Real Data
        scatter!(p2, 1:length(real_data), real_data, color=:black, markersize=3, label="Observed Data")

        savefig(p2, "plots/test3_uncertainty/$(state_name)_band.png")
    end
end

println("Uncertainty analysis complete. Check plots/test3_uncertainty/")
