# Test 7: Train RANDOM GRAPH topology
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines

rng = Random.default_rng()
Random.seed!(rng, 44)  # Different seed

TOPOLOGY = :random
println("=== Training: RANDOM GRAPH ===\n")

include("../../Train/preprocessing.jl")

latent_dim = 3
nin_tot = 1 + 3 + latent_dim
n_nodes = g.num_nodes

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
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    (x, (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4))
end

function sample_dropout_masks(model, st, x_shape)
    d1, d2, d3 = model.drop1, model.drop2, model.drop3
    mask1 = (rand(st.drop1.rng, Float32, (32, x_shape[2])) .> d1.p) ./ (1 - d1.p)
    mask2 = (rand(st.drop2.rng, Float32, (32, x_shape[2])) .> d2.p) ./ (1 - d2.p)
    mask3 = (rand(st.drop3.rng, Float32, (32, x_shape[2])) .> d3.p) ./ (1 - d3.p)
    (layer1=st.layer1, drop1=(mask=mask1, rng=st.drop1.rng),
        layer2=st.layer2, drop2=(mask=mask2, rng=st.drop2.rng),
        layer3=st.layer3, drop3=(mask=mask3, rng=st.drop3.rng), layer4=st.layer4)
end

# Create RANDOM graph (same density as original)
A_orig = Matrix(adjacency_matrix(g))
n_edges = sum(A_orig .> 0.01) - n_nodes
A_rand = zeros(Float64, n_nodes, n_nodes)
edges_added = 0
while edges_added < n_edges
    i, j = rand(1:n_nodes), rand(1:n_nodes)
    if i != j && A_rand[i, j] == 0.0
        w = rand()
        A_rand[i, j] = w
        A_rand[j, i] = w
        global edges_added += 2
    end
end
for i in 1:n_nodes
    A_rand[i, i] = 1.0
end
A_rand ./= maximum(A_rand)
g_train = GNNGraph(A_rand)

model = ExplicitGNN(nin_tot, 32, 1, 0.3)
ps, st = Lux.setup(rng, model)
latent_features = randn(Float32, latent_dim, n_nodes) .* 0.1
ps_full = ComponentArray(gnn=ps, latent_features=latent_features)

function loss_fn(p, stage_end)
    st_frozen = sample_dropout_masks(model, st, (nin_tot, n_nodes))
    function dudt_frozen(u, _, t)
        t_idx = clamp(searchsortedfirst(tsteps, t), 1, length(tsteps))
        cov_val = [covariate_splines[v, n](t) for v in 1:3, n in 1:n_nodes]
        input_mat = vcat(reshape(u, 1, n_nodes), cov_val, p.latent_features)
        y, _ = model(g_train, input_mat, p.gnn, st_frozen)
        vec(y)
    end
    prob = ODEProblem(dudt_frozen, u0, (0.0, Float64(stage_end)))
    sol = solve(prob, Tsit5(), saveat=tsteps[1:stage_end], reltol=1e-4, abstol=1e-4)
    sol.retcode != :Success && return Inf
    pred = reduce(hcat, sol.u)
    sum((pred .- X_norm[1, 1:stage_end, :]') .^ 2) / length(pred)
end

curriculum_steps = [10, 20, 40, 60, 90, 120, 150, 180]
epochs_per_stage = [50, 100, 150, 150, 150, 150, 150, 100]
lr_schedule = [5e-4, 2e-4, 2e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5]

loss_history = Float64[]

for (i_stage, stage_end) in enumerate(curriculum_steps)
    lr, nepochs = lr_schedule[i_stage], epochs_per_stage[i_stage]
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps_full)
    println("Stage $i_stage/$(length(curriculum_steps)): T=$stage_end, LR=$lr, Epochs=$nepochs")

    for epoch in 1:nepochs
        loss_val, grads = Zygote.withgradient(p -> loss_fn(p, stage_end), ps_full)
        !isfinite(loss_val) && (println("  Epoch $epoch: NaN"); continue)
        global ps_full
        opt_state, ps_full = Optimisers.update(opt_state, ps_full, grads[1])
        push!(loss_history, loss_val)

        if epoch % 20 == 0 || epoch == 1
            println("  Epoch $epoch: Loss = $(round(loss_val, digits=6))")
        end
    end
end

@save "Resultados/test-7/checkpoints/params_random.jld2" ps_full
CSV.write("Resultados/test-7/checkpoints/loss_random.csv", DataFrame(epoch=1:length(loss_history), loss=loss_history))
println("\n✅ RANDOM GRAPH COMPLETE: Final Loss = $(round(loss_history[end], digits=6))")
