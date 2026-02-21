# resume_phase5.jl
#
# Phase 5: Micro-Dropout Perturbation + Deterministic Annealing
# Resumes from the definitive Phase 4 checkpoint.
# Introduces a tiny amount of dropout (e.g., 2%) for the first cycles to perturb 
# the model out of its local minimum, then switches back to 0.0 dropout for 
# a final deterministic annealing descent.

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, SciMLSensitivity
using Zygote, Optimization, OptimizationOptimisers
using LinearAlgebra, Statistics, Random, Plots, CubicSplines, NPZ, Printf

rng = Random.default_rng()
Random.seed!(rng, 123)

println("=== PHASE 5: MICRO-DROPOUT PERTURBATION & DETERMINISTIC DESCEND ===\n")

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
# We modify FrozenDropout so we can toggle the dropout rate and mask dynamically.
mutable struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end

Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, d::FrozenDropout) = (mask=nothing, rng=rng, p=d.p)

function (d::FrozenDropout)(x, ps, st)
    # If p is 0.0 or mask is disabled, pass through
    if st.p == 0.0 || st.mask === nothing
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

# Initialize with 0.0 dropout, we will manually inject masks
gnn = ExplicitGNN(nin_tot, 64, 1, 0.0)
_, st_gnn = Lux.setup(rng, gnn)

# ─── 3. Load Phase 4 Checkpoint ──────────────────────────────────────────────
ckpt_path = "Resultados/test-8BIS/checkpoints/params_40s_phase4_ext.jld2"
println("Loading Phase 4 definitive checkpoint: params_40s_phase4_ext.jld2")
ps_raw = jldopen(ckpt_path, "r") do f
    f["ps_final"]
end
ps_init = ComponentArray(Float32.(ps_raw))
println("Transfer complete.")

# ─── 4. Training Core ────────────────────────────────────────────────────────
# Helper to set dynamic dropout masks
function update_dropout_mask!(st, p_drop, rng, feature_dim, num_nodes)
    if p_drop == 0.0
        # Disable dropout
        st = (
            layer1=st.layer1, drop1=(mask=nothing, rng=st.drop1.rng, p=0.0),
            layer2=st.layer2, drop2=(mask=nothing, rng=st.drop2.rng, p=0.0),
            layer3=st.layer3, drop3=(mask=nothing, rng=st.drop3.rng, p=0.0),
            layer4=st.layer4
        )
    else
        # Enable dropout: Create identical mask across batch, scale by 1/(1-p)
        scale = 1.0f0 / (1.0f0 - Float32(p_drop))
        mask1 = rand(rng, Float32, feature_dim, 1) .> Float32(p_drop)
        mask1 = Float32.(mask1) .* scale
        mask1_full = repeat(mask1, 1, num_nodes)

        mask2 = rand(rng, Float32, feature_dim, 1) .> Float32(p_drop)
        mask2 = Float32.(mask2) .* scale
        mask2_full = repeat(mask2, 1, num_nodes)

        mask3 = rand(rng, Float32, feature_dim, 1) .> Float32(p_drop)
        mask3 = Float32.(mask3) .* scale
        mask3_full = repeat(mask3, 1, num_nodes)

        st = (
            layer1=st.layer1, drop1=(mask=mask1_full, rng=st.drop1.rng, p=p_drop),
            layer2=st.layer2, drop2=(mask=mask2_full, rng=st.drop2.rng, p=p_drop),
            layer3=st.layer3, drop3=(mask=mask3_full, rng=st.drop3.rng, p=p_drop),
            layer4=st.layer4
        )
    end
    return st
end

global current_st = st_gnn

function dudt(u, p, t)
    u_r = reshape(u, 1, n_nodes)
    cov = map(s -> s(t), splines_train)
    lat = p.latent_features
    y, _ = gnn(g_train, vcat(u_r, cov, lat), p.gnn, current_st)
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

epochs_per_cycle = 40
cycles_dropout = 2      # Cycles to run with micro-dropout
cycles_deterministic = 3 # Cycles to run completely deterministically
total_epochs = (cycles_dropout + cycles_deterministic) * epochs_per_cycle
micro_drop_rate = 0.02  # 2% dropout

scheduler = CosineAnnealing(1e-5, 2e-7, epochs_per_cycle)

optz = Optimisers.setup(Optimisers.Adam(scheduler.η_max), ps_init)
current_ps = deepcopy(ps_init)

global best_mae = 999.0f0
global mae_history = Float64[]
mkpath("Resultados/test-8BIS/checkpoints")

println("\nStarting Phase 5: Micro-Dropout & Deterministic Final Descent")
println("Total Epochs: $(total_epochs) ($(cycles_dropout) Dropout, $(cycles_deterministic) Deterministic)")
println("Micro-Dropout Rate: $(micro_drop_rate)")
println("LR Range: $(scheduler.η_max) -> $(scheduler.η_min)")

global_epoch = 0

for c in 1:(cycles_dropout+cycles_deterministic)
    is_dropout_phase = c <= cycles_dropout
    phase_name = is_dropout_phase ? "Micro-Dropout (2%)" : "Deterministic (0%)"
    println("\n--- Cycle $c - $phase_name ---")

    for ep in 1:epochs_per_cycle
        global global_epoch += 1

        lr = get_lr(scheduler, ep - 1)
        Optimisers.adjust!(optz, lr)

        t0 = time()

        # Determine dropout for the forward pass (training phase)
        if is_dropout_phase
            global current_st = update_dropout_mask!(current_st, micro_drop_rate, rng, 64, n_nodes)
        else
            global current_st = update_dropout_mask!(current_st, 0.0, rng, 64, n_nodes)
        end

        # Calculate gradients using the current (potentially dropped out) state
        grads = Zygote.gradient(p -> loss_global(p), current_ps)[1]

        # Update parameters
        Optimisers.update!(optz, current_ps, grads)

        # For evaluation, always compute deterministic MAE regardless of phase
        global current_st = update_dropout_mask!(current_st, 0.0, rng, 64, n_nodes)
        eval_mae = loss_global(current_ps)

        dt = time() - t0
        push!(mae_history, eval_mae)

        @printf("Cycle %d Ep %d/%d (Global %d/%d) | Eval MAE = %.5f | LR = %.2e | %.1fs\n",
            c, ep, epochs_per_cycle, global_epoch, total_epochs, eval_mae, lr, dt)

        # Checkpoint if best (always measured deterministically)
        if eval_mae < best_mae
            global best_mae = eval_mae
            jldsave("Resultados/test-8BIS/checkpoints/params_40s_phase5_final.jld2"; ps_final=current_ps)
            println("  > Phase 5 Checkpoint saved (best MAE = $(round(eval_mae, digits=5)))")

            p = plot(mae_history, label="Deterministic MAE", xlabel="Epoch", ylabel="Loss", title="Phase 5: Perturbation & Annealing")
            savefig(p, "Resultados/test-8BIS/plots/phase5_loss.png")
            println("  > Plot saved")
        end
    end
end

println("\nPhase 5 Training Complete!")
println("Best Deterministic MAE achieved: $(round(best_mae, digits=5))")
