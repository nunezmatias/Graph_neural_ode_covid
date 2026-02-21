# resume_phase6.jl
#
# Phase 6: Extended Deterministic Final Descent
# Resumes from the definitive Phase 4 checkpoint (params_40s_phase4_ext.jld2).
# No dropout, strictly deterministic, running for ~2 hours to see if we can
# squeeze out an even deeper minimum.

using Pkg
Pkg.activate(".")
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, SciMLSensitivity
using Zygote, Optimization, OptimizationOptimisers
using LinearAlgebra, Statistics, Random, Plots, CubicSplines, NPZ, Printf

rng = Random.default_rng()
Random.seed!(rng, 123)

println("=== PHASE 6: EXTENDED DETERMINISTIC DESCEND (FROM PHASE 4) ===\n")

# ─── 1. Load Data ────────────────────────────────────────────────────────────
holdout_states = ["AZ", "LA", "MA", "MD", "NM", "NV", "RI", "TN", "UT"]
train_states = [
    "AL", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TX", "VA", "VT", "WA", "WI", "WV", "WY"
]
n_nodes = length(train_states)

features_raw = NPZ.npzread("Data/data_filtered.npz")
n_vars, n_times = size(features_raw["NY"])
tsteps = Float32.(collect(0:n_times-1))

X_train = zeros(Float32, n_vars, n_times, n_nodes)
for (i, s) in enumerate(train_states)
    X_train[:, :, i] = features_raw[s]
end
X_train_norm = log.(X_train .+ 1.0f0)

splines_train = [CubicSpline(tsteps, @view X_train_norm[v, :, n]) for v in 2:size(X_train_norm, 1), n in 1:n_nodes]

df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
col_names = names(df_adj)[2:end]

function build_graph_and_adj(states)
    idx = [findfirst(x -> x == s, col_names) for s in states]
    A = Matrix(df_adj[:, 2:end])[idx, idx]
    A ./= maximum(A)
    for i in 1:length(states)
        A[i, i] = 0.0
    end
    return GNNGraph(A + I), A
end

g_train, adj_train = build_graph_and_adj(train_states)

# ─── 2. Model Definition ─────────────────────────────────────────────────────
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

function ExplicitGNN(nin, nhid, nout, dp)
    ExplicitGNN(
        GNNLux.GraphConv(nin => nhid, tanh; aggr=mean), FrozenDropout(dp),
        GNNLux.GraphConv(nhid => nhid, tanh; aggr=mean), FrozenDropout(dp),
        GNNLux.GraphConv(nhid => nhid, tanh; aggr=mean), FrozenDropout(dp),
        GNNLux.GraphConv(nhid => nout; aggr=mean)
    )
end

Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialparameters(rng, m.layer1), drop1=NamedTuple(),
    layer2=Lux.initialparameters(rng, m.layer2), drop2=NamedTuple(),
    layer3=Lux.initialparameters(rng, m.layer3), drop3=NamedTuple(),
    layer4=Lux.initialparameters(rng, m.layer4),
)

Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1),
    layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2),
    layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3),
    layer4=Lux.initialstates(rng, m.layer4),
)

function (m::ExplicitGNN)(g, x, ps, st)
    x, s1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, sd1 = m.drop1(x, ps.drop1, st.drop1)
    x = typeof(x) <: Tuple ? x[1] : x

    x, s2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, sd2 = m.drop2(x, ps.drop2, st.drop2)
    x = typeof(x) <: Tuple ? x[1] : x

    x, s3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, sd3 = m.drop3(x, ps.drop3, st.drop3)
    x = typeof(x) <: Tuple ? x[1] : x

    x, s4 = m.layer4(g, x, ps.layer4, st.layer4)

    return x, (layer1=s1, drop1=sd1, layer2=s2, drop2=sd2, layer3=s3, drop3=sd3, layer4=s4)
end

nin_covar = size(splines_train, 1)
nin_tot = 1 + nin_covar + 3
gnn = ExplicitGNN(nin_tot, 64, 1, 0.0) # 0.0 dropout natively
_, st_gnn = Lux.setup(rng, gnn)

model_state = st_gnn

# ─── 3. Load Phase 4 Checkpoint ──────────────────────────────────────────────
ckpt_path = "Resultados/test-8BIS/checkpoints/params_40s_phase4_ext.jld2"
println("Loading Phase 4 checkpoint: params_40s_phase4_ext.jld2")
ps_raw = jldopen(ckpt_path, "r") do f
    f["ps_final"]
end
ps_init = ComponentArray(Float32.(ps_raw))
println("Transfer complete.")

# ─── 4. Training Core ────────────────────────────────────────────────────────
function dudt(u, p, t)
    u_r = reshape(u, 1, n_nodes)
    cov = map(s -> s(t), splines_train)
    lat = p.latent_features
    y, _ = gnn(g_train, vcat(u_r, cov, lat), p.gnn, model_state)
    return vec(y)
end

target_full = X_train_norm[1:1, :, :]

function loss_global(ps)
    u0 = X_train_norm[1, 1, :]
    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]), ps)
    sol = solve(prob, Tsit5(), saveat=tsteps, reltol=1e-3, abstol=1e-3,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

    pred = reshape(reduce(hcat, sol.u), 1, n_nodes, length(tsteps))
    pred = permutedims(pred, (1, 3, 2))

    gt = target_full
    return mean(abs.(pred .- gt))
end

# ─── 5. Custom Cosine Annealing Loop ─────────────────────────────────────────

struct CosineAnnealing
    η_max::Float64
    η_min::Float64
    T_cycle::Int
end

function get_lr(opt::CosineAnnealing, epoch_in_cycle::Int)
    return opt.η_min + 0.5 * (opt.η_max - opt.η_min) * (1 + cos(pi * epoch_in_cycle / opt.T_cycle))
end

# To get ~2 hours, assuming 40 seconds per epoch: 2 hrs = 7200s, 7200/40 = 180 epochs.
# We'll do 5 cycles of 36 epochs = 180 epochs.
epochs_per_cycle = 36
cycles = 5
total_epochs = cycles * epochs_per_cycle

# Very fine learning rate tuning since we are already at the minimum
scheduler = CosineAnnealing(5e-6, 1e-7, epochs_per_cycle)

optz = Optimisers.setup(Optimisers.Adam(scheduler.η_max), ps_init)
current_ps = deepcopy(ps_init)

# Initialize best_mae by measuring the initial state
global best_mae = loss_global(current_ps)
println("Initial MAE from Phase 4 checkpoint: $(round(best_mae, digits=5))")

global mae_history = Float64[best_mae]
mkpath("Resultados/test-8BIS/checkpoints")

println("\nStarting Phase 6 Extended Deterministic Training")
println("Total Epochs: $(total_epochs) ($cycles cycles of $epochs_per_cycle epochs)")
println("LR Range: $(scheduler.η_max) -> $(scheduler.η_min)")

global_epoch = 0

for c in 1:cycles
    println("\n--- Cycle $c ---")
    for ep in 1:epochs_per_cycle
        global global_epoch += 1

        lr = get_lr(scheduler, ep - 1)
        Optimisers.adjust!(optz, lr)

        t0 = time()

        # Calculate gradients
        grads = Zygote.gradient(p -> loss_global(p), current_ps)[1]

        # Update parameters
        Optimisers.update!(optz, current_ps, grads)

        # Evaluate deterministic MAE
        mae = loss_global(current_ps)

        dt = time() - t0
        push!(mae_history, mae)

        @printf("Cycle %d Ep %d/%d (Global %d/%d) | MAE = %.5f | LR = %.2e | %.1fs\n",
            c, ep, epochs_per_cycle, global_epoch, total_epochs, mae, lr, dt)

        # Checkpoint if best
        if mae < best_mae
            global best_mae = mae
            jldsave("Resultados/test-8BIS/checkpoints/params_40s_phase6_final.jld2"; ps_final=current_ps, loss_history=mae_history)
            println("  > Phase 6 Checkpoint saved (best MAE = $(round(mae, digits=5)))")

            p = plot(mae_history, label="MAE", xlabel="Epoch", ylabel="Loss", title="Phase 6 Extended Training (No Dropout)")
            savefig(p, "Resultados/test-8BIS/plots/phase6_loss.png")
            # println("  > Plot saved")
        end
        flush(stdout)
    end
end

println("\nPhase 6 Training Complete!")
println("Best MAE achieved: $(round(best_mae, digits=5))")
