# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using JSON
using NPZ

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("=== Test 8 BIS: FAST TRAINING (40 STATES) ===")
println("Using ", Threads.nthreads(), " threads")

# -------------------------------------------------------------
# 1. State Selection (from Test 9)
# -------------------------------------------------------------
# Selected in Test 9 structurally to preserve connectivity while holding out 3 from each cluster
holdout_states = ["AZ", "LA", "MA", "MD", "NM", "NV", "RI", "TN", "UT"]
train_states = [
    "AL", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TX", "VA", "VT", "WA", "WI", "WV", "WY"
]

n_nodes = length(train_states)
println("Training Nodes: ", n_nodes)
println("Holdout Nodes (Ignored for now): ", length(holdout_states))

# -------------------------------------------------------------
# 2. Data Loading & Feature Extraction
# -------------------------------------------------------------
println("Loading specific 40 train states from NPZ...")
features_raw = NPZ.npzread("Data/data_filtered.npz")

# Extract only Train states
example_feat = features_raw["NY"]
n_vars, n_times = size(example_feat)

X_tensor = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(train_states)
    X_tensor[:, :, i] = features_raw[state]
end

# Log normalization for all features
X_norm = log.(X_tensor .+ 1.0)
tsteps = Float32.(collect(0:n_times-1))

# Splines (Variables 2:4)
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

# Adjacency matrix formatting
df_adj_full = CSV.read("Data/adj_pop_dist.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
indices = [findfirst(x -> x == s, col_names) for s in train_states]

adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

using SparseArrays

# Max-Only Normalization
maxA = maximum(adj_sub)
adj_norm = adj_sub ./ maxA
adj_norm[adj_norm.<0.05] .= 0.0 # Sparsify connections below 5%
for i in 1:n_nodes
    adj_norm[i, i] = 0.0
end
adj_sparse = sparse(adj_norm)
adj_final = adj_sparse + I # Add self loops
g = GNNGraph(adj_final)

# -------------------------------------------------------------
# 3. Model Definition (Width 64)
# -------------------------------------------------------------
latent_dim = 3
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

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

println("Initializing GNN (Width 64)...")
gnn = ExplicitGNN(nin_tot, 64, nout, 0.0)
_, st_gnn = Lux.setup(rng, gnn)

# -------------------------------------------------------------
# 4. Transfer Learning (Warm Start)
# -------------------------------------------------------------
save_path = "Resultados/test-8BIS/checkpoints/params_40s_fast.jld2"

if isfile(save_path)
    println("Resuming from existing Test-8BIS checkpoint: $save_path")
    @load save_path ps_final
    ps = ps_final
else
    println("Loading weights from Test 8 (25 states) for Warm Start...")
    @load "Resultados/test-8/checkpoints/params_test8_25s.jld2" ps_trained

    # Extract Latent distribution from the 25 trained nodes
    old_latents = ps_trained.latent_features
    μ_lat = mean(old_latents)
    σ_lat = std(old_latents)

    # Generate new latents for 40 nodes from same distribution
    new_latents = (randn(rng, Float32, latent_dim, n_nodes) .* σ_lat) .+ μ_lat

    # Merge into new parameters wrapper
    ps = ComponentArray(gnn=ps_trained.gnn, latent_features=new_latents) |> f64
end

# -------------------------------------------------------------
# 5. Temporal Chunking Training Logic
# -------------------------------------------------------------
# ODE Forward pass
function predict_chunk(p, t_start_idx, t_len, st, spls)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        cov_matrix = map(s -> s(t), spls)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = gnn(g, model_input, p_ode.gnn, st)
        return vec(y)
    end

    t_chunk = Float32.(collect(t_start_idx:t_start_idx+t_len-1))
    u0 = X_norm[1, t_start_idx+1, :] # +1 because Julia is 1-indexed, time is 0-indexed

    prob = ODEProblem(dudt, vec(u0), (t_chunk[1], t_chunk[end]))

    # We use InterpolatingAdjoint because it's faster and more stable for short chunks than Backsolve
    sol = solve(prob, Tsit5(), p=p, saveat=t_chunk,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        reltol=1e-3, abstol=1e-3) # Relaxed tolerances for speed!

    return sol, t_chunk
end

function loss_chunk(p, t_start_idx, t_len, st, spls)
    sol, t_chunk = predict_chunk(p, t_start_idx, t_len, st, spls)

    if sol.retcode != :Success || length(sol.t) != t_len
        return 9999.0, st, NamedTuple()
    end

    sol_matrix = reshape(reduce(hcat, sol.u), nin_target, n_nodes, t_len)
    pred = permutedims(sol_matrix, (1, 3, 2))

    # Target slice (+1 for 1-based index)
    target = X_norm[1:1, t_start_idx+1:t_start_idx+t_len, :]

    # MAE for heavy outliers like California
    loss = mean(abs, target .- pred)
    return loss, st, NamedTuple()
end

# -------------------------------------------------------------
# 6. Optimized Training Loop
# -------------------------------------------------------------
opt = Optimisers.AdamW(eta=1e-4, lambda=1e-4) # Start with low LR since it's fine-tuning
tstate = Lux.Training.TrainState(gnn, ps, st_gnn, opt)

println("\n--- Starting Phase 1: Temporal Curriculum Chunking ---")
# Extended curriculum
n_iterations = 2000

loss_hist = []

function moving_average(vs, n)
    [sum(@view vs[max(1, i - n + 1):i]) / min(n, i) for i in 1:length(vs)]
end

for i in 1:n_iterations
    # Temporal Curriculum Logic
    if i <= 500
        chunk_size = 15
    elseif i <= 1000
        chunk_size = 30
    else
        chunk_size = 45
    end

    # Pick a random start day ensuring we stay within the 400 days bounds
    t_start = rand(0:(n_times-chunk_size-1))

    # Optional LR decay
    if i == 500
        Optimisers.adjust!(tstate.optimizer_state, 5e-5)
        println("  > Decreased Learning Rate to 5e-5")
    elseif i == 1500
        Optimisers.adjust!(tstate.optimizer_state, 1e-5)
        println("  > Decreased Learning Rate to 1e-5")
    end

    loss_val = 0.0
    function loss_f(m, p_try, s_try, d)
        l, s_new, _ = loss_chunk(p_try, t_start, chunk_size, s_try, covariate_splines)
        loss_val = l # Capture for printing
        return l, s_new, NamedTuple()
    end

    global tstate
    grads, l, _, tstate = Training.single_train_step!(
        AutoZygote(), loss_f, nothing, tstate
    )

    push!(loss_hist, l)

    # Print real-time
    if i % 10 == 0 || i == 1
        println("Iter $i | StartDay=$t_start | Chunk=$chunk_size | ChunkLoss(MAE) = $(round(loss_val, digits=5))")
    end

    if i % 100 == 0 || i == n_iterations
        ma_loss = moving_average(loss_hist, 10)
        p_loss = plot(loss_hist, label="ChunkLoss (MAE)", xlabel="Iteración", ylabel="MAE", title="Phase 1 - Temporal Curriculum", linewidth=1, color=:steelblue, size=(800, 400))
        plot!(p_loss, ma_loss, label="Media móvil (10)", linewidth=2, color=:darkorange)
        if i >= 500
            vline!(p_loss, [500], label="Horizon=30 (LR 5e-5)", linestyle=:dash, color=:steelblue)
        end
        if i >= 1000
            vline!(p_loss, [1000], label="Horizon=45", linestyle=:dash, color=:darkred)
        end
        if i >= 1500
            vline!(p_loss, [1500], label="LR 1e-5", linestyle=:dash, color=:purple)
        end
        min_loss, min_idx = findmin(loss_hist)
        scatter!(p_loss, [min_idx], [min_loss], label="Mínimo: $(round(min_loss, digits=5)) (iter $min_idx)", color=:steelblue)
        savefig(p_loss, "Resultados/test-8BIS/plots/chunk_loss_history_iter_$i.png")
        if i == n_iterations
            savefig(p_loss, "Resultados/test-8BIS/plots/chunk_loss_history.png")
            println("Saved final Phase 1 Curriculum Loss plot to Resultados/test-8BIS/plots/chunk_loss_history.png")
        else
            println("Saved intermediate Phase 1 Chunk Loss plot to Resultados/test-8BIS/plots/chunk_loss_history_iter_$i.png")
        end
    end
end
println("Saved Phase 1 Chunk Loss plot to Resultados/test-8BIS/plots/chunk_loss_history.png")

println("\n--- Starting Phase 2: Global Synchronization ---")
# Run a few epochs over the full 400 days with strict tolerances to stitch predictions
Optimisers.adjust!(tstate.optimizer_state, 1e-5)

function predict_full(p, st)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        cov_matrix = map(s -> s(t), covariate_splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = gnn(g, model_input, p_ode.gnn, st)
        return vec(y)
    end
    u0 = X_norm[1, 1, :]
    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    # Back to BacksolveAdjoint for global stab or Interpolating if memory allows. 
    # Let's use Backsolve for safety on 400 steps
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps,
        sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
        reltol=1e-4, abstol=1e-5)
    return sol
end

# We will use the already Zygote-traced loss_curriculum from Phase 1, but with full length
global_epochs = 30
full_horizon = length(tsteps)
for ep in 1:global_epochs
    loss_val_global = 0.0
    function loss_fg(m, p_try, s_try, d)
        # Using Phase 1's loss_curriculum which correctly tracks gradients!
        l, s_new, _ = loss_curriculum(m, p_try, s_try, d, 1, full_horizon)
        loss_val_global = l
        return l, s_new, NamedTuple()
    end

    global grads, l, _, tstate = Training.single_train_step!(
        AutoZygote(), loss_fg, nothing, tstate
    )
    println("Global Epoch $ep/$global_epochs | Global Loss = $(round(loss_val_global, digits=5))")
end

save_path = "Resultados/test-8BIS/checkpoints/params_40s_fast.jld2"
println("Saving calibrated parameters to $save_path")
ps_final = tstate.parameters
@save save_path ps_final

# Quick debug plot of the first state (AL)
sol_final = predict_full(ps_final, st_gnn)
pred_al = [u[1] for u in sol_final.u]
real_al = X_norm[1, :, 1]
p = plot(title="Training Final Plot (400 Days) - Node 1 (AL)")
plot!(p, tsteps, real_al, label="Real Log Cases", color=:black, alpha=0.5)
plot!(p, tsteps, pred_al, label="Pred Global", color=:red)
savefig(p, "Resultados/test-8BIS/plots/debug_train_AL.png")

println("Done! Time: ", time() - strt_time)
