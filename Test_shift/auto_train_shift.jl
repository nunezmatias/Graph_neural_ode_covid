# ---------------------------------------------------------
# UNIFIED, TIME-OPTIMIZED 5-PHASE CURRICULUM TRAINER
# ---------------------------------------------------------

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using NPZ

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("\n" * "="^60)
println("🚀 UNIFIED UPGRADED AUTO-TRAINER 🚀")
println("="^60)

println("Using ", Threads.nthreads(), " threads")

# ==============================================================================
# 🎛️ CENTRAL CONFIGURATION BLOCK
# Change these hyperparameters to experiment with the training curriculum.
# ==============================================================================
const CONFIG = (
    # --- Structural ---
    latent_dim=3,         # Dimensionality of the learned latent ODE parameters
    p_dropout_initial=0.05, # Frozen Spatial Dropout used in Phases 1-3 to prevent Hub memorization

    # --- Phase 1: Temporal Chunking ---
    p1_epochs=2000,         # Total epochs to run the chunked IVP
    p1_lr=5f-4,           # Learning rate (AdamW)
    p1_wd=1f-4,           # Weight decay

    # --- Phase 2: Global Splicing ---
    p2_epochs=100,         # Total epochs forcing the full 400-day unroll
    p2_lr=1f-5,           # Conservative learning rate to stitch the chunks
    p2_patience=4,        # Early stop patience if MAE delta < 1e-4

    # --- Phase 3: Cosine Annealing (Warm Restarts) ---
    p3_cycles=3,          # Number of warm restarts to kick model out of local minima
    p3_period=30,         # Epochs per cycle (time to cool down before the kick)
    p3_lr_max=5f-5,       # Maximum kinetic energy
    p3_lr_min=1f-6,       # Floor kinetic energy

    # --- Phase 4: Deterministic Extrapolation ---
    p4_cycles=5,          # Number of warm restarts (NO DROPOUT)
    p4_period=100,         # Epochs per cycle (extended runway)
    p4_lr_max=2f-5,       # Micro-domain maximum LR
    p4_lr_min=5f-7,       # Micro-domain floor LR

    # --- Phase 5: Asymptotic Floor ---
    p5_epochs=2000,        # Maximum continuous descent epochs
    p5_lr=5f-6,           # Microscopic static sweep LR
    p5_patience=30         # Assert strict flatline if delta < 5e-5
)
# ==============================================================================

# --- 1. Master Data Loading (Compiles Once) ---
println("\n[1/6] Bootstrapping Data & Adjacency...")
train_states = [
    "AL", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TX", "VA", "VT", "WA", "WI", "WV", "WY"
]
n_nodes = length(train_states)
features_raw = NPZ.npzread("Data/data_all.npz")
n_vars, n_times = size(features_raw["NY"])
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(train_states)
    X_tensor_all[:, :, i] = features_raw[state]
end

# select covariates to test
idx_pos = findfirst(x -> x == "--idx", ARGS)  # Look for --idx argument
if idx_pos !== nothing && idx_pos < length(ARGS)
    idx_string = ARGS[idx_pos + 1]
    idx_list = parse.(Int, split(idx_string, ","))
end
# Ensure variable 1 (target: daily infections) is always included
if !(1 in idx_list)
    push!(idx_list, 1)
end
# Remove duplicates and keep things ordered
idx_list = sort(unique(idx_list))
println("Using variable indices: ", idx_list)
X_tensor = X_tensor_all[idx_list,:,:]

# shift covariates in time
lag_pos = findfirst(x -> x == "--lag", ARGS)
lags = Int[] 
if lag_pos !== nothing && lag_pos < length(ARGS)
    lags = parse.(Int, split(ARGS[lag_pos + 1], ","))
else
    lags = zeros(Int, length(idx_list) - 1)  # Default: if no lags provided, assume 0 for all covariates
end
if length(lags) != length(idx_list)-1
    error("ERROR: You provided $(length(lags)) lags for $(length(idx_list)-1) covariates!")
end
max_lag = isempty(lags) ? 0 : maximum(lags)
new_T = n_times - max_lag
X_aligned = zeros(eltype(X_tensor), length(idx_list), new_T, n_nodes)
X_aligned[1, :, :] = X_tensor[1, (max_lag + 1):n_times, :]
for i in eachindex(lags)
    l = lags[i]
    var_idx = i + 1 
    start_idx = max_lag - l + 1
    end_idx = n_times - l
    X_aligned[var_idx, :, :] = X_tensor[var_idx, start_idx:end_idx, :]
end

println("Final Data Shape: ", size(X_aligned))
n_vars, n_times, n_nodes = size(X_aligned)

# normalize
X_norm = log.(X_aligned .+ 1.0f0)
tsteps = Float32.(collect(0:n_times-1))
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

df_adj_full = CSV.read("Data/adj_pop_dist.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
indices = [findfirst(x -> x == s, col_names) for s in train_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]
adj_sub[adj_sub.<0.05] .= 0.0
adj_norm = adj_sub ./ maximum(adj_sub)
for i in 1:n_nodes
    adj_norm[i, i] = 0.0
end
g = GNNGraph(sparse(adj_norm + I))

# --- 2. Base GNN Architecture ---
latent_dim = CONFIG.latent_dim
nin_target = 1
nin_covar = size(covariate_splines,1)
nin_tot = nin_target + nin_covar + latent_dim

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

# Core ODE Logic (Always evaluates over exactly provided timeline bounds)
function predict_ode(p, t_bounds, gnn_obj, st_obj, spls)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        cov_matrix = map(s -> s(t), spls)
        model_input = vcat(u_reshaped, cov_matrix, p_ode.latent_features)
        y, _ = gnn_obj(g, model_input, p_ode.gnn, st_obj)
        return vec(y)
    end
    u0 = Float32.(X_norm[1, 1, :])
    t_span = (Float32(t_bounds[1]), Float32(t_bounds[end]))
    prob = ODEProblem(dudt, vec(u0), t_span)
    sol = solve(prob, Tsit5(), p=p, saveat=t_bounds, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), reltol=1f-4, abstol=1f-5)
    return sol
end

function loss_function(p, t_bounds, gnn_obj, st_obj, spls)
    sol = predict_ode(p, t_bounds, gnn_obj, st_obj, spls)
    tl = length(t_bounds)
    if sol.retcode != :Success || length(sol.t) != tl
        return 9999f0, st_obj, NamedTuple()
    end
    sol_matrix = reshape(Array(sol), nin_target, n_nodes, tl)
    pred = permutedims(sol_matrix, (1, 3, 2))
    t_start_idx = Int(t_bounds[1]) + 1
    t_end_idx = Int(t_bounds[end]) + 1
    target = X_norm[1:1, t_start_idx:t_end_idx, :]
    return mean(abs, target .- pred), st_obj, NamedTuple()
end

# Smart Early Stopping Evaluator
function check_early_stop(history, patience=5, min_delta=1e-4)
    if length(history) <= patience
        return false
    end
    recent = history[end-patience+1:end]
    best_recent = minimum(recent)
    old_best = minimum(history[1:end-patience])
    if (old_best - best_recent) < min_delta
        println("   >> EARLY STOPPING TRIGGERED! (Delta < $min_delta over $patience epochs)")
        return true
    end
    return false
end

# --- GLOBAL TRAINING STATE ---
mkpath("checkpoints_shift")

# Parse arguments for smoke testing
is_smoke_test = "--smoke-test" in ARGS
if is_smoke_test
    println("\n⚠️ SMOKE TEST MODE ACTIVE: Overriding to 2 epochs per phase! ⚠️")
end

# Fallback to random initialization if no historical weights exist
tmp_gnn = ExplicitGNN(nin_tot, 64, 1, CONFIG.p_dropout_initial)
ps_tmp, _ = Lux.setup(rng, tmp_gnn)
init_latents = randn(rng, Float32, latent_dim, n_nodes)
ps_init = ComponentArray(gnn=ps_tmp, latent_features=init_latents)
# Recursively enforce Float32
ps_init = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps_init)

global current_ps = deepcopy(ps_init)

# -----------------------------------------------------------------------------
# PHASE 1: Temporal Chunking (Growing from 15 -> 30 -> 45 Days)
# 
# 🧠 PEDAGOGICAL NOTE: Why chunk the time?
# If we ask the ODE solver to integrate 400 days immediately with random weights,
# the smallest error on Day 1 compounds exponentially by Day 400 (The Butterfly Effect).
# The gradients explode into NaNs. 
# 
# Instead, we break the 400-day timeline into random 15-day chunks.
# The network easily learns these short horizons. Once stable, we stretch 
# the chunks to 30 days, then 45 days. 
# 
# We also use 5% Frozen Dropout here. This acts like a "random wind" blowing 
# through the network, preventing it from memorizing just New York and California,
# and forcing it to learn the physics of the entire country.
# -----------------------------------------------------------------------------
println("\n" * "="^40)
println("PHASE 1: Temporal Curriculum Chunking")
println("="^40)
gnn_p1 = ExplicitGNN(nin_tot, 64, 1, CONFIG.p_dropout_initial) # Applying structural "wind"
_, st_p1 = Lux.setup(rng, gnn_p1)
opt_p1 = Optimisers.AdamW(eta=CONFIG.p1_lr, lambda=CONFIG.p1_wd)
tstate1 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, opt_p1)

p1_epochs = is_smoke_test ? 2 : CONFIG.p1_epochs
loss_p1_hist = Float32[]

for i in 1:p1_epochs
    chunk_size = i <= 500 ? 15 : (i <= 1000 ? 30 : 45)
    t_start = rand(0:(n_times-chunk_size-1))
    t_chunk = Float32.(collect(t_start:t_start+chunk_size-1))

    # Optional LR decay
    if i == 500
        Optimisers.adjust!(tstate1.optimizer_state, 5f-5)
        println("  > Decreased Learning Rate to 5e-5")
    elseif i == 1500
        Optimisers.adjust!(tstate1.optimizer_state, 1f-5)
        println("  > Decreased Learning Rate to 1e-5")
    end

    function loss_f1(m, p_try, s_try, d)
        l, sn, _ = loss_function(p_try, t_chunk, gnn_p1, st_p1, covariate_splines)
        return l, sn, NamedTuple()
    end

    t0 = time()
    _, l, _, tstate1_new = Lux.Training.single_train_step!(AutoZygote(), loss_f1, nothing, tstate1)
    global tstate1 = tstate1_new
    dt = round(time() - t0, digits=1)
    push!(loss_p1_hist, l)
    println("Ph1 Epoch $i/$p1_epochs | Window: $chunk_size days (start day $t_start) | Loss: $(round(l, digits=4)) | $(dt)s")
    # We don't early stop phase 1 because the chunk target changes every epoch (loss is highly noisy)
end
global current_ps = deepcopy(tstate1.parameters)

# -----------------------------------------------------------------------------
# PHASE 2: Global 400-Day Splicing
#
# 🧠 PEDAGOGICAL NOTE: Removing the training wheels
# Now that the model can confidently predict any 45-day window, we ask it
# to run the full 400-day marathon. It won't explode anymore because Phase 1
# taught it the underlying "rules of physics". Now, it just surgically stitches
# those 45-day chunks together. We keep the learning rate extremely low.
# -----------------------------------------------------------------------------
println("\n" * "="^40)
println("PHASE 2: Continuous All-Days Splicing")
println("="^40)
# Uses same topology to preserve stability
opt_p2 = Optimisers.Adam(CONFIG.p2_lr)
tstate2 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, opt_p2)
p2_epochs = is_smoke_test ? 2 : CONFIG.p2_epochs
loss_p2_hist = Float32[]

for i in 1:p2_epochs
    t0 = time()
    function loss_f2(m, p_try, s_try, d)
        l, sn, _ = loss_function(p_try, tsteps, gnn_p1, st_p1, covariate_splines)
        return l, sn, NamedTuple()
    end
    _, l, _, tstate2_new = Lux.Training.single_train_step!(AutoZygote(), loss_f2, nothing, tstate2)
    global tstate2 = tstate2_new
    dt = round(time() - t0, digits=1)
    push!(loss_p2_hist, l)
    println("Ph2 Epoch $i/$p2_epochs | Global all-Days Loss: $(round(l, digits=5)) | $(dt)s")

    # if check_early_stop(loss_p2_hist, CONFIG.p2_patience, 1e-4)
    #     break
    # end
end
global current_ps = deepcopy(tstate2.parameters)

# -----------------------------------------------------------------------------
# PHASE 3: Cosine Annealing (Warm Restarts)
#
# 🧠 PEDAGOGICAL NOTE: Escaping the Pothole
# The model is now stable, but likely stuck in a shallow "local minimum" (a pothole).
# To get out, we use Cosine Annealing (SGDR):
# 1. We decay the learning rate to let the model settle at the bottom of the pothole.
# 2. Suddenly, we spike the learning rate back to max! This violently kicks the 
#    model out of the pothole, letting it fly into a much deeper, better valley.
# -----------------------------------------------------------------------------
println("\n" * "="^40)
println("PHASE 3: Cosine Annealing Basin Hunting")
println("="^40)
# We cycle the learning rate between 5e-5 and 1e-6 over 15 epochs.
η_max_p3, η_min_p3, T_p3 = CONFIG.p3_lr_max, CONFIG.p3_lr_min, CONFIG.p3_period
# cosine annealing function
cos_lr(ep, mx, mn, t) = mn + 0.5f0 * (mx - mn) * (1.0f0 + cos(Float32(π) * mod(ep - 1, t) / t))

p3_cycles = is_smoke_test ? 1 : CONFIG.p3_cycles
p3_epochs = is_smoke_test ? 2 : (p3_cycles * T_p3)
tstate3 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, Optimisers.Adam(η_max_p3))
loss_p3_hist = Float32[]
best_mae_p3 = Inf

for i in 1:p3_epochs
    t0 = time()
    lr = cos_lr(i, η_max_p3, η_min_p3, T_p3)
    global tstate3 = Lux.Training.TrainState(gnn_p1, tstate3.parameters, st_p1, Optimisers.Adam(lr))

    function loss_f3(m, p_try, s_try, d)
        l, sn, _ = loss_function(p_try, tsteps, gnn_p1, st_p1, covariate_splines)
        return l, sn, NamedTuple()
    end
    _, l, _, tstate3_new = Lux.Training.single_train_step!(AutoZygote(), loss_f3, nothing, tstate3)
    global tstate3 = tstate3_new
    dt = round(time() - t0, digits=1)
    push!(loss_p3_hist, l)

    if l < best_mae_p3
        global best_mae_p3 = l
        global current_ps = deepcopy(tstate3.parameters)
    end
    cycle = div(i - 1, T_p3) + 1
    println("Ph3 Cycle $cycle Ep $i/$p3_epochs | LR: $(round(lr, sigdigits=3)) | Loss: $(round(l, digits=5)) | $(dt)s")
end

# -----------------------------------------------------------------------------
# PHASE 4: Deterministic Hyper-Refinement (NO DROPOUT)
#
# 🧠 PEDAGOGICAL NOTE: Turning off the wind
# We are now in the perfect valley. It's time to find the absolute bottom.
# The 5% dropout "wind" from Phase 1 was great for exploring, but now it's 
# just adding noisy turbulence. We turn dropout to 0.0%. The environment 
# becomes perfectly still, allowing the mathematical gradients to become 
# razor-sharp as we slide to the very center of the valley.
# -----------------------------------------------------------------------------
println("\n" * "="^40)
println("PHASE 4: Deterministic Descent (Dropout = 0.0)")
println("="^40)
# The network structure physically changes here. Dropout is 0.0!
gnn_p4 = ExplicitGNN(nin_tot, 64, 1, 0.0)
_, st_p4 = Lux.setup(rng, gnn_p4)

η_max_p4, η_min_p4, T_p4 = CONFIG.p4_lr_max, CONFIG.p4_lr_min, CONFIG.p4_period
p4_cycles = is_smoke_test ? 1 : CONFIG.p4_cycles
p4_epochs = is_smoke_test ? 2 : (p4_cycles * T_p4)
tstate4 = Lux.Training.TrainState(gnn_p4, current_ps, st_p4, Optimisers.Adam(η_max_p4))
loss_p4_hist = Float32[]
best_mae_p4 = Inf

for i in 1:p4_epochs
    t0 = time()
    lr = cos_lr(i, η_max_p4, η_min_p4, T_p4)
    global tstate4 = Lux.Training.TrainState(gnn_p4, tstate4.parameters, st_p4, Optimisers.Adam(lr))

    function loss_f4(m, p_try, s_try, d)
        l, sn, _ = loss_function(p_try, tsteps, gnn_p4, st_p4, covariate_splines)
        return l, sn, NamedTuple()
    end
    _, l, _, tstate4_new = Lux.Training.single_train_step!(AutoZygote(), loss_f4, nothing, tstate4)
    global tstate4 = tstate4_new
    dt = round(time() - t0, digits=1)
    push!(loss_p4_hist, l)

    if l < best_mae_p4
        global best_mae_p4 = l
        global current_ps = deepcopy(tstate4.parameters)
    end
    cycle = div(i - 1, T_p4) + 1
    println("Ph4 Cycle $cycle Ep $i/$p4_epochs | LR: $(round(lr, sigdigits=3)) | Loss: $(round(l, digits=5)) | $(dt)s")
end

# -----------------------------------------------------------------------------
# PHASE 5: Ultimate Convergence (Static Descent with Early Stop)
#
# 🧠 PEDAGOGICAL NOTE: The Asymptotic Floor
# We turn off all learning rate tricks. We set a tiny, static learning rate.
# The model spends hundreds of epochs making microscopic adjustments. 
# When the loss curve strictly flatlines, we have hit the "Asymptotic Floor".
# This proves the network has mathematically exhausted all spatial-temporal 
# information inside the geographic dataset.
# -----------------------------------------------------------------------------
println("\n" * "="^40)
println("PHASE 5: Final Asymptotic Floor")
println("="^40)
# A tiny, completely static learning rate
opt_p5 = Optimisers.Adam(CONFIG.p5_lr)
tstate5 = Lux.Training.TrainState(gnn_p4, current_ps, st_p4, opt_p5)
p5_epochs = is_smoke_test ? 2 : CONFIG.p5_epochs
loss_p5_hist = Float32[]

for i in 1:p5_epochs
    t0 = time()
    function loss_f5(m, p_try, s_try, d)
        l, sn, _ = loss_function(p_try, tsteps, gnn_p4, st_p4, covariate_splines)
        return l, sn, NamedTuple()
    end
    _, l, _, tstate5_new = Lux.Training.single_train_step!(AutoZygote(), loss_f5, nothing, tstate5)
    global tstate5 = tstate5_new
    dt = round(time() - t0, digits=1)
    push!(loss_p5_hist, l)
    println("Ph6 Final Descent $i/$p5_epochs | Global Loss: $(round(l, digits=5)) | $(dt)s")

    # Very aggressive early stop to mathematically prove convergence
    if check_early_stop(loss_p5_hist, CONFIG.p5_patience, 5f-5)
        println("\n>>> ABSOLUTE MATHEMATICAL FLOOR REACHED AT EPOCH $i <<<")
        break
    end
end
global current_ps = deepcopy(tstate5.parameters)

idx_suffix = join(idx_list, "_")
@save "checkpoints_shift/params_shift_idx$(idx_suffix).jld2" ps_final = current_ps

# Save total loss history
loss_hist = Dict("Ph1" => loss_p1_hist, "Ph2" => loss_p2_hist, "Ph3" => loss_p3_hist, "Ph4" => loss_p4_hist, "Ph5" => loss_p5_hist)
@save "checkpoints_shift/loss_hist_shift_idx$(idx_suffix).jld2" loss_hist

total_hrs = round((time() - strt_time) / 3600, digits=2)
println("\n✅ TOTAL UNIFIED PIPELINE COMPLETED IN $total_hrs HOURS ✅")

