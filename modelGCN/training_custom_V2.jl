
# ==============================================================================
# 🎛️ CENTRAL CONFIGURATION & STRUCTS 
# ==============================================================================
const CONFIG = (
    latent_dim = 3,
    p_dropout_initial = 0.05,
    p1_epochs = 2000,   p1_lr = 5f-4,     p1_wd = 1f-4,
    p2_epochs = 100,    p2_lr = 1f-5,     p2_patience = 4,
    p3_cycles = 3,      p3_period = 30,   p3_lr_max = 5f-5,   p3_lr_min = 1f-6,
    p4_cycles = 5,      p4_period = 100,  p4_lr_max = 2f-5,   p4_lr_min = 5f-7,
    p5_epochs = 3000,   p5_lr = 5f-6,     p5_patience = 30
)

# Frozen Dropout Layer
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end

Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    if st.mask === nothing
        return x, st
    else
        return (x .* st.mask) / (1 - d.p), st 
    end
end

# NEW LAYER

struct WeightedGraphLayer{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    n_nodes::Int      # NEW: needed to size the per-node bias
    act::F
end

WeightedGraphLayer(in_dims::Int, out_dims::Int, n_nodes::Int; act=identity) = 
    WeightedGraphLayer(in_dims, out_dims, n_nodes, act)

function Lux.initialparameters(rng::Random.AbstractRNG, l::WeightedGraphLayer)
    (W = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
     b = zeros(Float32, l.out_dims, l.n_nodes))   # was (out_dims, 1) -> now per-node
end

Lux.initialstates(::Random.AbstractRNG, ::WeightedGraphLayer) = NamedTuple()

function (l::WeightedGraphLayer)(x::AbstractMatrix, A::AbstractMatrix, ps, st)
    x_agg = x * A
    out = l.act.(ps.W * x_agg .+ ps.b)  
    return out, st
end

# NEW ARCHITECTURE 

struct ExplicitGNN{L1,D1,L2,D2,L3,D3,L4} <: Lux.AbstractLuxLayer
    layer1::L1
    drop1::D1
    layer2::L2
    drop2::D2
    layer3::L3
    drop3::D3
    layer4::L4
end

function ExplicitGNN(nin, nhidden, nout, n_nodes, drop_p)
    return ExplicitGNN(
        WeightedGraphLayer(nin, nhidden, n_nodes; act=tanh),
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden, n_nodes; act=tanh),
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden, n_nodes; act=tanh),
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nout, n_nodes)   
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

# Forward pass: A instead of g
function (layer::ExplicitGNN)(A::AbstractMatrix, x::AbstractMatrix, ps, st)
    x1, st_c1 = layer.layer1(x, A, ps.layer1, st.layer1)
    x1, st_d1 = layer.drop1(x1, ps.drop1, st.drop1)

    x2, st_c2 = layer.layer2(x1, A, ps.layer2, st.layer2)
    x2, st_d2 = layer.drop2(x2, ps.drop2, st.drop2)
    x2 = x2 .* x1

    x3, st_c3 = layer.layer3(x2, A, ps.layer3, st.layer3)
    x3, st_d3 = layer.drop3(x3, ps.drop3, st.drop3)
    x3 = x3 .+ x2

    x4, st_c4 = layer.layer4(x3, A, ps.layer4, st.layer4)

    new_st = (layer1=st_c1, drop1=st_d1, layer2=st_c2, drop2=st_d2, 
              layer3=st_c3, drop3=st_d3, layer4=st_c4)
    return x4, new_st
end

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

cos_lr(ep, mx, mn, t) = mn + 0.5f0 * (mx - mn) * (1.0f0 + cos(Float32(π) * mod(ep - 1, t) / t))

# ==============================================================================
# 🚀 CORE SINGLE TRAINING FUNCTION
# ==============================================================================
function train_model(X_norm::Array{Float32, 3}, A::Matrix{Float32}, tsteps::Vector{Float32}, spls::Matrix, save_path::String; seed::Int=123)
    strt_time = time()
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    println("\n" * "="^60)
    println("🚀 SINGLE RUN TRAINING | SEED $seed 🚀")
    println("="^60)

    n_vars, n_times, n_nodes = size(X_norm)
    latent_dim = CONFIG.latent_dim
    nin_target = 1
    nin_covar = size(spls, 1)
    nin_tot = nin_target + nin_covar + latent_dim

    # ODE & Loss Functions
    function predict_ode(p, t_bounds, gnn_obj, st_obj)
        function dudt(u, p_ode, t)
            u_reshaped = reshape(u, nin_target, n_nodes)
            cov_matrix = map(s -> s(t), spls)
            model_input = vcat(u_reshaped, cov_matrix, p_ode.latent_features)
            y, _ = gnn_obj(A, model_input, p_ode.gnn, st_obj)
            return vec(y)
        end
        u0 = Float32.(X_norm[1, 1, :])
        t_span = (Float32(t_bounds[1]), Float32(t_bounds[end]))
        prob = ODEProblem(dudt, vec(u0), t_span)
        sol = solve(prob, Tsit5(), p=p, saveat=t_bounds, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), reltol=1f-4, abstol=1f-5)
        return sol
    end

    function loss_function(p, t_bounds, gnn_obj, st_obj)
        sol = predict_ode(p, t_bounds, gnn_obj, st_obj)
        tl = length(t_bounds)
        if sol.retcode != ReturnCode.Success || length(sol.t) != tl
            return 9999f0, st_obj, NamedTuple()
        end
        sol_matrix = reshape(Array(sol), nin_target, n_nodes, tl)
        pred = permutedims(sol_matrix, (1, 3, 2))
        t_start_idx = Int(t_bounds[1]) + 1
        t_end_idx = Int(t_bounds[end]) + 1
        target = X_norm[1:1, t_start_idx:t_end_idx, :]
        return mean(abs, target .- pred), st_obj, NamedTuple()
    end

    # Model Initialization
    tmp_gnn = ExplicitGNN(nin_tot, 64, 1, CONFIG.p_dropout_initial)
    ps_tmp, _ = Lux.setup(rng, tmp_gnn)
    init_latents = randn(rng, Float32, latent_dim, n_nodes)
    ps_init = ComponentArray(gnn=ps_tmp, latent_features=init_latents)
    current_ps = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, ps_init)

    # ==========================================================================
    # PHASE 1: Temporal Chunking
    # ==========================================================================
    println("\n--- PHASE 1: Temporal Chunking ---")
    gnn_p1 = ExplicitGNN(nin_tot, 64, 1, CONFIG.p_dropout_initial)
    _, st_p1 = Lux.setup(rng, gnn_p1)
    tstate1 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, Optimisers.AdamW(eta=CONFIG.p1_lr, lambda=CONFIG.p1_wd))
    loss_p1_hist = Float32[]

    for i in 1:CONFIG.p1_epochs
        chunk_size = i <= 500 ? 15 : (i <= 1000 ? 30 : 45)
        t_start = rand(rng, 0:(n_times-chunk_size-1))
        t_chunk = Float32.(collect(t_start:t_start+chunk_size-1))

        if i == 500
            Optimisers.adjust!(tstate1.optimizer_state, 5f-5)
        elseif i == 1500
            Optimisers.adjust!(tstate1.optimizer_state, 1f-5)
        end

        function loss_f1(m, p_try, s_try, d)
            l, sn, _ = loss_function(p_try, t_chunk, gnn_p1, st_p1)
            return l, sn, NamedTuple()
        end

        t0 = time()
        _, l, _, tstate1_new = Lux.Training.single_train_step!(AutoZygote(), loss_f1, nothing, tstate1)
        tstate1 = tstate1_new
        dt = round(time() - t0, digits=1)
        push!(loss_p1_hist, l)
    end
    current_ps = deepcopy(tstate1.parameters)

    # ==========================================================================
    # PHASE 2: Global 400-Day Splicing
    # ==========================================================================
    println("\n--- PHASE 2: Continuous All-Days Splicing ---")
    tstate2 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, Optimisers.Adam(CONFIG.p2_lr))
    loss_p2_hist = Float32[]

    for i in 1:CONFIG.p2_epochs
        t0 = time()
        function loss_f2(m, p_try, s_try, d)
            l, sn, _ = loss_function(p_try, tsteps, gnn_p1, st_p1)
            return l, sn, NamedTuple()
        end
        _, l, _, tstate2_new = Lux.Training.single_train_step!(AutoZygote(), loss_f2, nothing, tstate2)
        tstate2 = tstate2_new
        dt = round(time() - t0, digits=1)
        push!(loss_p2_hist, l)
    end
    current_ps = deepcopy(tstate2.parameters)

    # ==========================================================================
    # PHASE 3: Cosine Annealing Basin Hunting
    # ==========================================================================
    println("\n--- PHASE 3: Cosine Annealing Basin Hunting ---")
    p3_epochs = CONFIG.p3_cycles * CONFIG.p3_period
    tstate3 = Lux.Training.TrainState(gnn_p1, current_ps, st_p1, Optimisers.Adam(CONFIG.p3_lr_max))
    loss_p3_hist = Float32[]
    best_mae_p3 = Inf

    for i in 1:p3_epochs
        t0 = time()
        lr = cos_lr(i, CONFIG.p3_lr_max, CONFIG.p3_lr_min, CONFIG.p3_period)
        tstate3 = Lux.Training.TrainState(gnn_p1, tstate3.parameters, st_p1, Optimisers.Adam(lr))

        function loss_f3(m, p_try, s_try, d)
            l, sn, _ = loss_function(p_try, tsteps, gnn_p1, st_p1)
            return l, sn, NamedTuple()
        end
        _, l, _, tstate3_new = Lux.Training.single_train_step!(AutoZygote(), loss_f3, nothing, tstate3)
        tstate3 = tstate3_new
        dt = round(time() - t0, digits=1)
        push!(loss_p3_hist, l)

        if l < best_mae_p3
            best_mae_p3 = l
            current_ps = deepcopy(tstate3.parameters)
        end
    end

    # ==========================================================================
    # PHASE 4: Deterministic Descent (Dropout = 0.0)
    # ==========================================================================
    println("\n--- PHASE 4: Deterministic Descent ---")
    gnn_p4 = ExplicitGNN(nin_tot, 64, 1, 0.0)
    _, st_p4 = Lux.setup(rng, gnn_p4)

    p4_epochs = CONFIG.p4_cycles * CONFIG.p4_period
    tstate4 = Lux.Training.TrainState(gnn_p4, current_ps, st_p4, Optimisers.Adam(CONFIG.p4_lr_max))
    loss_p4_hist = Float32[]
    best_mae_p4 = Inf

    for i in 1:p4_epochs
        t0 = time()
        lr = cos_lr(i, CONFIG.p4_lr_max, CONFIG.p4_lr_min, CONFIG.p4_period)
        tstate4 = Lux.Training.TrainState(gnn_p4, tstate4.parameters, st_p4, Optimisers.Adam(lr))

        function loss_f4(m, p_try, s_try, d)
            l, sn, _ = loss_function(p_try, tsteps, gnn_p4, st_p4)
            return l, sn, NamedTuple()
        end
        _, l, _, tstate4_new = Lux.Training.single_train_step!(AutoZygote(), loss_f4, nothing, tstate4)
        tstate4 = tstate4_new
        dt = round(time() - t0, digits=1)
        push!(loss_p4_hist, l)

        if l < best_mae_p4
            best_mae_p4 = l
            current_ps = deepcopy(tstate4.parameters)
        end
    end

    # ==========================================================================
    # PHASE 5: Ultimate Convergence
    # ==========================================================================
    println("\n--- PHASE 5: Final Asymptotic Floor ---")
    tstate5 = Lux.Training.TrainState(gnn_p4, current_ps, st_p4, Optimisers.Adam(CONFIG.p5_lr))
    loss_p5_hist = Float32[]

    for i in 1:CONFIG.p5_epochs
        t0 = time()
        function loss_f5(m, p_try, s_try, d)
            l, sn, _ = loss_function(p_try, tsteps, gnn_p4, st_p4)
            return l, sn, NamedTuple()
        end
        _, l, _, tstate5_new = Lux.Training.single_train_step!(AutoZygote(), loss_f5, nothing, tstate5)
        tstate5 = tstate5_new
        dt = round(time() - t0, digits=1)
        push!(loss_p5_hist, l)
        
        if i % 10 == 0
            println("Ph5 Final Descent $i/$(CONFIG.p5_epochs) | Loss: $(round(l, digits=5)) | $(dt)s")
        end

        if check_early_stop(loss_p5_hist, CONFIG.p5_patience, 5f-5)
            println("\n>>> FLOOR REACHED AT EPOCH $i <<<")
            break
        end
    end
    current_ps = deepcopy(tstate5.parameters)

    # ==========================================================================
    # SAVE
    # ==========================================================================
    param_file = joinpath(save_path, "params_customV2.jld2")
    loss_file = joinpath(save_path, "lossHist_customV2.jld2")
    
    @save param_file ps_final = current_ps
    
    loss_hist = Dict("Ph1" => loss_p1_hist, "Ph2" => loss_p2_hist, "Ph3" => loss_p3_hist, "Ph4" => loss_p4_hist, "Ph5" => loss_p5_hist)
    @save loss_file loss_hist
    
    total_hrs = round((time() - strt_time) / 3600, digits=2)
    println("\n✅ TRAINING COMPLETED IN $total_hrs HOURS ✅")
    
    return true
end