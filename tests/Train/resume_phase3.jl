# resume_phase3.jl - Test 8BIS: Cosine Annealing Full-Horizon Training
# Resumes from Phase 2 checkpoint (params_40s_phase2.jld2) and applies
# cosine annealing with warm restarts for LR cycling.

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using NPZ

Training = Lux.Training
strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("=== PHASE 3 (Cosine Annealing): Full 400-Day Training ===")

# --- 1. Data ---
train_states = [
    "AL", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TX", "VA", "VT", "WA", "WI", "WV", "WY"
]
n_nodes = length(train_states)
features_raw = NPZ.npzread("Data/data_filtered.npz")
n_vars, n_times = size(features_raw["NY"])
X_tensor = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(train_states)
    X_tensor[:, :, i] = features_raw[state]
end
X_norm = log.(X_tensor .+ 1.0f0)
tsteps = Float32.(collect(0:n_times-1))
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

df_adj_full = CSV.read("Data/adj_pop_dist.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
indices = [findfirst(x -> x == s, col_names) for s in train_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]
adj_sub[adj_sub.<0.05] .= 0
adj_norm = adj_sub ./ maximum(adj_sub)
for i in 1:n_nodes
    adj_norm[i, i] = 0.0
end
g = GNNGraph(sparse(adj_norm + I))

# --- 2. Model ---
latent_dim = 3;
nin_target = 1;
nin_covar = 3;
nin_tot = nin_target + nin_covar + latent_dim;

struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)
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
    ExplicitGNN(
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end
Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialparameters(rng, m.layer1), drop1=NamedTuple(),
    layer2=Lux.initialparameters(rng, m.layer2), drop2=NamedTuple(),
    layer3=Lux.initialparameters(rng, m.layer3), drop3=NamedTuple(),
    layer4=Lux.initialparameters(rng, m.layer4)
)
Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1),
    layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2),
    layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3),
    layer4=Lux.initialstates(rng, m.layer4)
)
function (m::ExplicitGNN)(g, x, ps, st)
    x, st_1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st_1, drop1=st_d1, layer2=st_2, drop2=st_d2, layer3=st_3, drop3=st_d3, layer4=st_4)
end

println("Initializing GNN (Width 64)...")
gnn = ExplicitGNN(nin_tot, 64, 1, 0.05)
ps_gnn, st_gnn = Lux.setup(rng, gnn)

# --- 3. Load Phase 2 Checkpoint ---
save_path_in = "Resultados/test-8BIS/checkpoints/params_40s_phase2.jld2"
save_path_out = "Resultados/test-8BIS/checkpoints/params_40s_phase3.jld2"
@load save_path_in ps_final
ps = ComponentArray(Float32.(ps_final))
println("Phase 2 checkpoint loaded. MAE will continue from here.")

# --- 4. Loss (full 400-day horizon) ---
function predict_full(p, spls)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        cov_matrix = map(s -> s(t), spls)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = gnn(g, model_input, p_ode.gnn, st_gnn)
        return vec(y)
    end
    t_full = Float32.(collect(0:n_times-1))
    u0 = X_norm[1, 1, :]
    prob = ODEProblem(dudt, vec(u0), (t_full[1], t_full[end]))
    sol = solve(prob, Tsit5(), p=p, saveat=t_full,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        reltol=1f-3, abstol=1f-3)
    return sol, t_full
end

function loss_full(p, spls)
    sol, t_full = predict_full(p, spls)
    tl = length(t_full)
    if sol.retcode != :Success || length(sol.t) != tl
        return 9999f0, st_gnn, NamedTuple()
    end
    sol_matrix = reshape(reduce(hcat, sol.u), nin_target, n_nodes, tl)
    pred = permutedims(sol_matrix, (1, 3, 2))
    target = X_norm[1:1, 1:tl, :]
    return mean(abs, target .- pred), st_gnn, NamedTuple()
end

# --- 5. Cosine Annealing Schedule ---
# 3 warm restart cycles: each cycle lasts T_period epochs
# LR oscillates between η_max and η_min within each cycle.
η_max = 5f-5    # warm restart peak (moderate — we're already converged)
η_min = 1f-6    # cooldown minimum
n_cycles = 3       # number of warm restarts
T_period = 30      # epochs per cycle
n_epochs = n_cycles * T_period   # total epochs = 90

function cosine_lr(epoch)
    # Which cycle are we in?
    t_cur = mod(epoch - 1, T_period)
    return η_min + 0.5f0 * (η_max - η_min) * (1.0f0 + cos(Float32(π) * t_cur / T_period))
end

println("\n--- Cosine Annealing Schedule ---")
println("  η_max = $η_max, η_min = $η_min")
println("  $n_cycles cycles × $T_period epochs = $n_epochs total epochs")
println("  Expected time: ~$(round(n_epochs * 20 / 60, digits=1)) min\n")

# --- 6. Training Loop ---
# We manually adjust LR each epoch by creating a new TrainState
opt = Optimisers.Adam(cosine_lr(1))
global tstate = Training.TrainState(gnn, ps, st_gnn, opt)

loss_hist = Float64[]
lr_hist = Float64[]
best_mae = Inf
best_ps = nothing

for ep in 1:n_epochs
    t0 = time()
    lr = cosine_lr(ep)
    global best_mae, best_ps

    # Re-create optimizer with new LR each epoch
    new_opt = Optimisers.Adam(lr)
    global tstate = Training.TrainState(gnn, tstate.parameters, st_gnn, new_opt)

    loss_val = 0f0
    function loss_f(m, p_try, s_try, d)
        l, sn, _ = loss_full(p_try, covariate_splines)
        loss_val = l
        return l, sn, NamedTuple()
    end

    grads, l, _, tstate = Training.single_train_step!(
        AutoZygote(), loss_f, nothing, tstate
    )
    push!(loss_hist, l)
    push!(lr_hist, lr)
    dt = round(time() - t0, digits=1)

    cycle = div(ep - 1, T_period) + 1
    cycle_ep = mod(ep - 1, T_period) + 1
    println("Cycle $cycle Ep $cycle_ep/$T_period (Global $ep/$n_epochs) | MAE = $(round(l, digits=5)) | LR = $(round(lr, sigdigits=3)) | $(dt)s")

    if l < best_mae
        best_mae = l
        best_ps = deepcopy(tstate.parameters)
    end

    if ep % 5 == 0 || ep == n_epochs
        # Save best checkpoint
        global ps_final = best_ps
        @save save_path_out ps_final
        println("  > Checkpoint saved (best MAE = $(round(best_mae, digits=5)))")

        # Plot
        p1 = plot(layout=(2, 1), size=(900, 600))
        plot!(p1[1], loss_hist, label="MAE", xlabel="Epoch", ylabel="MAE",
            title="Phase 3 Cosine Annealing", lw=2, color=:steelblue)
        hline!(p1[1], [best_mae], label="Best: $(round(best_mae, digits=5))", ls=:dash, color=:red)
        plot!(p1[2], lr_hist, label="Learning Rate", xlabel="Epoch", ylabel="LR",
            lw=2, color=:orange, yscale=:log10)
        savefig(p1, "Resultados/test-8BIS/plots/phase3_cosine_loss.png")
        println("  > Plot saved")
    end
end

println("\nPhase 3 (Cosine Annealing) Complete!")
println("Final MAE: $(round(loss_hist[end], digits=5))")
println("Best MAE: $(round(best_mae, digits=5))")
println("Total time: $(round(time() - strt_time, digits=1))s")
