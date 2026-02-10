using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays;
using Graphs, Lux, GNNLux;
using DifferentialEquations, DiffEqFlux;
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

# 1. SETUP & DATA
rng = Random.default_rng()
Random.seed!(rng, 42)
include("../../Train/preprocessing.jl")

# TOPOLOGY MODIFICATION: RANDOM
function create_random_graph(g_orig)
    Random.seed!(42)
    A_orig = Matrix(adjacency_matrix(g_orig))
    n_nodes = g_orig.num_nodes
    n_edges_target = sum(A_orig .> 0.01) - n_nodes
    A_rand = zeros(Float64, n_nodes, n_nodes)
    ecount = 0
    while ecount < n_edges_target
        i, j = rand(1:n_nodes), rand(1:n_nodes)
        if i != j && A_rand[i, j] == 0.0
            w = rand()
            A_rand[i, j] = w
            A_rand[j, i] = w
            ecount += 2
        end
    end
    for i in 1:n_nodes
        A_rand[i, i] = 1.0
    end
    A_rand ./= (maximum(A_rand) + 1e-9)
    return GNNGraph(A_rand)
end
g = create_random_graph(g)

# 2. MODEL DEFINITION
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
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end
# Overloads
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
function (m::ExplicitGNN)(g, x, ps, st)
    x, st1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st2 = m.drop1(x, ps.drop1, st.drop1)
    x, st3 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st4 = m.drop2(x, ps.drop2, st.drop2)
    x, st5 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st6 = m.drop3(x, ps.drop3, st.drop3)
    x, st7 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st1, drop1=st2, layer2=st3, drop2=st4, layer3=st5, drop3=st6, layer4=st7)
end
function sample_dropout_masks(model, st, x_shape)
    d1 = model.drop1
    m1 = rand(st.drop1.rng, Float32, (32, x_shape[2])) .> d1.p
    m1 = m1 ./ (1 - d1.p)
    st_d1 = (mask=m1, rng=st.drop1.rng)
    d2 = model.drop2
    m2 = rand(st.drop2.rng, Float32, (32, x_shape[2])) .> d2.p
    m2 = m2 ./ (1 - d2.p)
    st_d2 = (mask=m2, rng=st.drop2.rng)
    d3 = model.drop3
    m3 = rand(st.drop3.rng, Float32, (32, x_shape[2])) .> d3.p
    m3 = m3 ./ (1 - d3.p)
    st_d3 = (mask=m3, rng=st.drop3.rng)
    return (layer1=st.layer1, drop1=st_d1, layer2=st.layer2, drop2=st_d2, layer3=st.layer3, drop3=st_d3, layer4=st.layer4)
end

# 3. LOAD CHECKPOINT
checkpoint_path = "Resultados/test-7/checkpoints/params_random.jld2"
println("Loading $checkpoint_path...")
ps_trained = load(checkpoint_path, "ps_trained")
gnn = ExplicitGNN(7, 32, 1, 0.0) # Disabled dropout to match Test 2 baseline
_, st_gnn = Lux.setup(rng, gnn)

# 4. TRAINING LOGIC
function loss_curriculum(model, p, st, data, tsteps, splines)
    u0_data = data[1, 1, :]
    n_nodes = size(data, 3)
    st_frozen = sample_dropout_masks(model, st, (32, n_nodes))
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        cov = [splines[i, j](t) for i in 1:3, j in 1:10]
        lat = p_ode.latent_features
        model_input = vcat(u_reshaped, cov, lat)
        y, _ = model(g, model_input, p_ode.gnn, st_frozen)
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(u0_data), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), reltol=1e-5, abstol=1e-6)
    pred = permutedims(reshape(reduce(hcat, sol.u), 1, n_nodes, length(tsteps)), (1, 3, 2))
    return mean(abs2, data[1:1, :, :] .- pred), st, NamedTuple()
end

function train_fine_tune!(model, ps, st, opt, tsteps, X_data)
    tstate = Training.TrainState(model, ps, st, opt)
    for epoch in 1:1000
        grads, l, _, tstate = Training.single_train_step!(AutoZygote(), (m, p, s, d) -> loss_curriculum(m, p, s, d, tsteps, covariate_splines), X_data, tstate)
        println("Fine-Tune Stage | Epoch $epoch | Loss = $l")
        flush(stdout)
        if epoch % 50 == 0
            @save "Resultados/test-7/checkpoints/params_random_finetuned.jld2" ps_trained = tstate.parameters
        end
    end
    return tstate.parameters
end

# 5. EXECUTION
opt = Optimisers.AdamW(eta=1e-5, lambda=1.0f-4)
ps_final = train_fine_tune!(gnn, ps_trained, st_gnn, opt, tsteps, X_norm)
@save "Resultados/test-7/checkpoints/params_random_finetuned.jld2" ps_trained = ps_final
println("Fine-tuning Random Complete.")
