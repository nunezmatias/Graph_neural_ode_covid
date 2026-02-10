# Test 7: Graph Ablation Study - Training Script
# Trains three models with different graph topologies to quantify network contribution

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines

rng = Random.default_rng()
Random.seed!(rng, 42)

println("=== Test 7: Graph Ablation Study ===")
println("Running with ", Threads.nthreads(), " threads\n")

# Load preprocessing (shared across all models)
include("../../Train/preprocessing.jl")

println("Data loaded: ", size(X_norm))
println("Training on ", g.num_nodes, " nodes\n")

#################################################
# TOPOLOGY GENERATOR
#################################################

function create_topology(type::Symbol, A_original::Matrix{Float64})
    n = size(A_original, 1)

    if type == :full
        println("Creating FULL graph (real topology)")
        return A_original

    elseif type == :isolated
        println("Creating ISOLATED graph (diagonal only)")
        A_iso = zeros(Float64, n, n)
        for i in 1:n
            A_iso[i, i] = 1.0  # Self-loops only
        end
        return A_iso

    elseif type == :random
        println("Creating RANDOM graph (same density)")
        # Count edges in original (excluding diagonal)
        n_edges = sum(A_original .> 0.01) - n

        # Create random symmetric matrix
        A_rand = zeros(Float64, n, n)
        edges_added = 0

        while edges_added < n_edges
            i, j = rand(1:n), rand(1:n)
            if i != j && A_rand[i, j] == 0.0
                weight = rand()
                A_rand[i, j] = weight
                A_rand[j, i] = weight  # Symmetric
                edges_added += 2
            end
        end

        # Add self-loops and normalize
        for i in 1:n
            A_rand[i, i] = 1.0
        end

        # Normalize to [0, 1]
        max_val = maximum(A_rand)
        if max_val > 0
            A_rand ./= max_val
        end

        return A_rand
    else
        error("Unknown topology type: $type")
    end
end

#################################################
# MODEL DEFINITION (Test 2 Architecture)
#################################################

struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end

Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
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

Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialparameters(rng, m.layer1),
    drop1=NamedTuple(),
    layer2=Lux.initialparameters(rng, m.layer2),
    drop2=NamedTuple(),
    layer3=Lux.initialparameters(rng, m.layer3),
    drop3=NamedTuple(),
    layer4=Lux.initialparameters(rng, m.layer4)
)

Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialstates(rng, m.layer1),
    drop1=Lux.initialstates(rng, m.drop1),
    layer2=Lux.initialstates(rng, m.layer2),
    drop2=Lux.initialstates(rng, m.drop2),
    layer3=Lux.initialstates(rng, m.layer3),
    drop3=Lux.initialstates(rng, m.drop3),
    layer4=Lux.initialstates(rng, m.layer4)
)

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

# Test 2 Hyperparameters
latent_dim = 3
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

#################################################
# TRAINING FUNCTION
#################################################

function train_model(topology_type::Symbol, A_original::Matrix{Float64})
    println("\n" * "="^60)
    println("Training Model: ", uppercase(String(topology_type)))
    println("="^60)

    # Create topology-specific graph
    A_topo = create_topology(topology_type, A_original)
    g_topo = GNNGraph(A_topo)

    # Initialize model
    model = ExplicitGNN(nin_tot, 32, nout, 0.3)
    ps, st = Lux.setup(rng, model)

    # Initialize latent features
    latent_features = randn(Float32, latent_dim, n_nodes) .* 0.1

    # Combine parameters
    ps_full = ComponentArray(gnn=ps, latent_features=latent_features)

    # ODE Function
    function dudt(u, p, t)
        t_idx = searchsortedfirst(tsteps, t)
        t_idx = clamp(t_idx, 1, length(tsteps))

        cov_val = [covariate_splines[v, n](t) for v in 1:nin_covar, n in 1:n_nodes]
        u_reshaped = reshape(u, nin_target, n_nodes)
        input_mat = vcat(u_reshaped, cov_val, p.latent_features)

        y, _ = model(g_topo, input_mat, p.gnn, st)
        return vec(y)
    end

    # Loss function
    function loss_fn(p, stage_end)
        st_frozen = sample_dropout_masks(model, st, (nin_tot, n_nodes))

        function dudt_frozen(u, _, t)
            t_idx = searchsortedfirst(tsteps, t)
            t_idx = clamp(t_idx, 1, length(tsteps))

            cov_val = [covariate_splines[v, n](t) for v in 1:nin_covar, n in 1:n_nodes]
            u_reshaped = reshape(u, nin_target, n_nodes)
            input_mat = vcat(u_reshaped, cov_val, p.latent_features)

            y, _ = model(g_topo, input_mat, p.gnn, st_frozen)
            return vec(y)
        end

        prob = ODEProblem(dudt_frozen, u0, (0.0, Float64(stage_end)))
        sol = solve(prob, Tsit5(), saveat=tsteps[1:stage_end], reltol=1e-4, abstol=1e-4)

        if sol.retcode != :Success
            return Inf
        end

        pred = reduce(hcat, sol.u)
        target = X_norm[1, 1:stage_end, :]'

        return sum((pred .- target) .^ 2) / length(pred)
    end

    # Curriculum settings (Test 2)
    curriculum_steps = [10, 20, 30, 50, 75, 100, 125, 150, 180]
    epochs_per_stage = [50, 50, 100, 150, 200, 250, 300, 400, 1500]
    lr_schedule = [1e-3, 1e-3, 5e-4, 5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5]

    # Training loop
    loss_history = Float64[]

    for (i_stage, stage_end) in enumerate(curriculum_steps)
        lr = lr_schedule[i_stage]
        nepochs = epochs_per_stage[i_stage]
        opt = Optimisers.Adam(lr)
        opt_state = Optimisers.setup(opt, ps_full)

        println("\nStage $i_stage/$length(curriculum_steps): T_end=$stage_end, LR=$lr, Epochs=$nepochs")

        for epoch in 1:nepochs
            loss_val, grads = Zygote.withgradient(p -> loss_fn(p, stage_end), ps_full)

            if !isfinite(loss_val)
                println("  Epoch $epoch: Loss = NaN/Inf, skipping update")
                continue
            end

            opt_state, ps_full = Optimisers.update(opt_state, ps_full, grads[1])
            push!(loss_history, loss_val)

            if epoch % 50 == 0 || epoch == 1
                println("  Epoch $epoch: Loss = $(round(loss_val, digits=6))")
            end
        end
    end

    # Save results
    output_file = "Resultados/test-7/checkpoints/params_$(topology_type).jld2"
    @save output_file ps_full
    println("\nSaved checkpoint: $output_file")

    # Save loss history
    loss_file = "Resultados/test-7/checkpoints/loss_$(topology_type).csv"
    CSV.write(loss_file, DataFrame(epoch=1:length(loss_history), loss=loss_history))
    println("Saved loss history: $loss_file")

    return ps_full, loss_history
end

#################################################
# MAIN EXECUTION
#################################################

# Extract original adjacency from preprocessing and convert to dense
A_original = Matrix(adjacency_matrix(g))

# Train all three models
results = Dict()

for topology in [:full, :isolated, :random]
    ps_trained, loss_hist = train_model(topology, A_original)
    results[topology] = (params=ps_trained, loss=loss_hist)
end

println("\n" * "="^60)
println("ALL MODELS TRAINED")
println("="^60)
println("\nFinal Losses:")
for (topo, res) in results
    final_loss = res.loss[end]
    println("  $topo: $(round(final_loss, digits=6))")
end

println("\nCheckpoints saved in: Resultados/test-7/checkpoints/")
println("Next: Run compare_results.jl to generate analysis plots")
