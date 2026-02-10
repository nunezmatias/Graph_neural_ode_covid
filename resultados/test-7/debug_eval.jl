using JLD2, ComponentArrays, Lux, GNNLux, DifferentialEquations, Statistics, Random, CubicSplines, Plots

push!(LOAD_PATH, "/Users/matias/Documents/codigo/Graph_neural_ode_covid/Utils/GraphCreator/");
using GNNGraph_mod

# Model definition (MUST BE AT TOP LEVEL)
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)
function (d::FrozenDropout)(x, ps, st)
    return (x, st)
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
function (m::ExplicitGNN)(g, x, ps, st)
    x, _ = m.layer1(g, x, ps.layer1, st.layer1)
    x, _ = m.layer2(g, x, ps.layer2, st.layer2)
    x, _ = m.layer3(g, x, ps.layer3, st.layer3)
    x, _ = m.layer4(g, x, ps.layer4, st.layer4)
    return x, st
end

# Minimal test for get_prediction_full
function test_prediction()
    println("Loading parameters...")
    ps_trained = load("/Users/matias/Documents/codigo/Graph_neural_ode_covid/Resultados/test-7/checkpoints/params_full.jld2", "ps_trained")

    # Mock data for testing
    n_nodes = 10
    tsteps = collect(0.0:1.0:10.0)
    u0 = rand(n_nodes)

    # Mock splines
    splines = [CubicSpline(collect(0.0:10.0), rand(11)) for i in 1:3, j in 1:10]

    g_test = build_GNNGraph(rand(10, 10))

    model = ExplicitGNN(
        GNNLux.GraphConv(7 => 32, tanh; aggr=mean), FrozenDropout(0.0),
        GNNLux.GraphConv(32 => 32, tanh; aggr=mean), FrozenDropout(0.0),
        GNNLux.GraphConv(32 => 32, tanh; aggr=mean), FrozenDropout(0.0),
        GNNLux.GraphConv(32 => 1; aggr=mean)
    )

    gnn_ps = ps_trained.gnn
    latent_ps = ps_trained.latent_features
    st_eval = Lux.initialstates(Random.default_rng(), model)

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        t_clamped = clamp(t, 0.0, 10.0)
        cov_vals = [splines[i, j](t_clamped) for i in 1:3, j in 1:10]
        model_input = Float64.(vcat(u_reshaped, cov_vals, latent_ps))
        y, _ = model(g_test, model_input, gnn_ps, st_eval)
        return vec(y)
    end

    println("Testing solver...")
    prob = ODEProblem(dudt, u0, (0.0, 10.0))
    sol = try
        solve(prob, Tsit5(), saveat=tsteps)
    catch e
        println("ERROR IN SOLVE: ", e)
        return nothing
    end
    if sol !== nothing
        println("Success! Prediction size: ", size(reduce(hcat, sol.u)))
    end
end

test_prediction()
