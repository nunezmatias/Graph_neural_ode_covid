using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

# Setup environment
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

println("=== STARTING COMPREHENSIVE STUDY: TEST 7 ===")

# 1. LOAD DATA & PREPROCESSING
include("../../Train/preprocessing.jl")
X_full_raw = g.ndata.x
X_full_norm = Float64.(log.(X_full_raw .+ 1))

tsteps_full = collect(0.0:1.0:size(X_full_norm, 2)-1)
train_end_idx = 180
test_short_end_idx = 220

# Re-define splines using FULL DATA
full_splines = [CubicSpline(tsteps_full, @view X_full_norm[i, :, j]) for i in 2:4, j in 1:10]

# 2. LOG PARSING FUNCTION
function parse_loss_from_log(logfile)
    stages = Int[]
    epochs = Int[]
    losses = Float64[]
    if !isfile(logfile)
        return DataFrame(Stage=stages, Epoch=epochs, Loss=losses)
    end
    for line in eachline(logfile)
        m = match(r"Stage (\d+) \| Epoch (\d+) \| Loss = ([\d\.]+)", line)
        if m !== nothing
            push!(stages, parse(Int, m.captures[1]))
            push!(epochs, parse(Int, m.captures[2]))
            push!(losses, parse(Float64, m.captures[3]))
        end
    end
    return DataFrame(Stage=stages, Epoch=epochs, Loss=losses)
end

# 3. MODEL DEFINITION (EXACT COPY FROM TRAINING)
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

function Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialparameters(rng, m.layer1), drop1=Lux.initialparameters(rng, m.drop1),
        layer2=Lux.initialparameters(rng, m.layer2), drop2=Lux.initialparameters(rng, m.drop2),
        layer3=Lux.initialparameters(rng, m.layer3), drop3=Lux.initialparameters(rng, m.drop3),
        layer4=Lux.initialparameters(rng, m.layer4)
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1),
        layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2),
        layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3),
        layer4=Lux.initialstates(rng, m.layer4)
    )
end

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

function get_prediction_full(model, ps_trained, g_input, tsteps, splines, u0_full)
    n_nodes = g_input.num_nodes
    gnn_ps = ps_trained.gnn
    latent_ps = Float64.(ps_trained.latent_features)
    st_eval = Lux.initialstates(Random.default_rng(), model)

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        t_clamped = clamp(t, tsteps[1], tsteps[end])
        cov_vals = [splines[i, j](t_clamped) for i in 1:3, j in 1:10]
        model_input = Float64.(vcat(u_reshaped, cov_vals, latent_ps))
        y, _ = model(g_input, model_input, gnn_ps, st_eval)
        return vec(y)
    end

    prob = ODEProblem(dudt, Float64.(vec(u0_full)), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), saveat=tsteps, maxiters=1e5, reltol=1e-3, abstol=1e-3)
    return reduce(hcat, sol.u)
end

# 4. TOPOLOGY HELPERS
function create_isolated_graph(g_orig)
    A_iso = Matrix(1.0 * I, g_orig.num_nodes, g_orig.num_nodes)
    return GNNGraph(A_iso)
end

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

# 5. EXECUTION
gnn_eval = ExplicitGNN(7, 32, 1, 0.05)

cases = [
    ("Full", "Resultados/test-7/checkpoints/params_full.jld2", g, "Resultados/test-7/full_test7.log"),
    ("Isolated", "Resultados/test-7/checkpoints/params_isolated.jld2", create_isolated_graph(g), "Resultados/test-7/isolated_test7.log"),
    ("Random", "Resultados/test-7/checkpoints/params_random.jld2", create_random_graph(g), "Resultados/test-7/random_test7.log")
]

all_metrics = []
mkpath("Resultados/test-7/plots/study")

println("\n=== PLOTTING LOSS OVERLAPS ===")
p_comp = plot(title="Training Loss Comparison", xlabel="Step", ylabel="MSE (Log)", yscale=:log10, legend=:outerright)
for (name, _, _, log_path) in cases
    ldf = parse_loss_from_log(log_path)
    if !isempty(ldf)
        plot!(p_comp, ldf.Loss, label=name, alpha=0.6)
    end
end
savefig(p_comp, "Resultados/test-7/plots/study/loss_comparison_all.png")

for (name, param_path, g_topo, log_path) in cases
    println("\nProcessing case: $name")

    if isfile(param_path)
        ps_load = load(param_path, "ps_trained")
        u0_eval = X_full_norm[1, 1, :]

        println("Generating forecasts...")
        pred = get_prediction_full(gnn_eval, ps_load, g_topo, tsteps_full, full_splines, u0_eval)
        pred_t = pred'

        y_true = X_full_norm[1, :, :]

        err_train = mean((pred_t[1:train_end_idx, :] .- y_true[1:train_end_idx, :]) .^ 2)
        err_test_short = mean((pred_t[train_end_idx+1:test_short_end_idx, :] .- y_true[train_end_idx+1:test_short_end_idx, :]) .^ 2)
        err_test_long = mean((pred_t[test_short_end_idx+1:end, :] .- y_true[test_short_end_idx+1:end, :]) .^ 2)

        push!(all_metrics, (
            Model=name,
            Train_MSE=err_train,
            Short_Test_MSE=err_test_short,
            Long_Test_MSE=err_test_long
        ))

        for state_idx in [1, 5] # NY, IL
            p_pred = plot(title="Study: $name - $(states[state_idx])", xlabel="Days", ylabel="Log Cases")
            scatter!(p_pred, tsteps_full, y_true[:, state_idx], label="Data", color=:black, markersize=1, alpha=0.3)
            plot!(p_pred, tsteps_full, pred_t[:, state_idx], label="Prediction", color=:red, linewidth=2)
            vline!(p_pred, [train_end_idx], label="Train End", color=:blue, linestyle=:dash)
            savefig(p_pred, "Resultados/test-7/plots/study/pred_$(name)_$(states[state_idx]).png")
        end
    else
        println("Missing params for $name.")
    end
end

if !isempty(all_metrics)
    summary_df = DataFrame(all_metrics)
    println("\n=== SUMMARY OF RESULTS ===")
    println(summary_df)
    CSV.write("Resultados/test-7/study_results.csv", summary_df)
end

println("Study Complete.")
