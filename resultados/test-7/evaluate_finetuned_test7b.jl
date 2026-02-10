using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux, DifferentialEquations, DiffEqFlux, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

println("=== EVALUATING DEEP FINE-TUNING (TEST 7b) ===")

# 1. LOAD DATA & PREPROCESSING
include("../../Train/preprocessing.jl")
X_full_raw = g.ndata.x
X_full_norm = Float64.(log.(X_full_raw .+ 1))

tsteps_full = collect(0.0:1.0:size(X_full_norm, 2)-1)
train_end_idx = 180
test_short_end_idx = 220

# Re-define splines using FULL DATA
full_splines = [CubicSpline(tsteps_full, @view X_full_norm[i, :, j]) for i in 2:4, j in 1:10]

# 2. MODEL DEFINITION (Must match EXACTLY - Dropout=0.0)
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
    # Correct state initialization
    st_eval = (
        layer1=NamedTuple(), drop1=(mask=nothing, rng=Random.default_rng()),
        layer2=NamedTuple(), drop2=(mask=nothing, rng=Random.default_rng()),
        layer3=NamedTuple(), drop3=(mask=nothing, rng=Random.default_rng()),
        layer4=NamedTuple()
    )

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

# 3. TOPOLOGY HELPERS
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

# 4. EXECUTION
gnn_eval = ExplicitGNN(7, 32, 1, 0.0) # Match Fine-Tuning Dropout

cases = [
    ("Full_Finetuned", "Resultados/test-7/checkpoints/params_full_finetuned.jld2", g),
    ("Isolated_Finetuned", "Resultados/test-7/checkpoints/params_isolated_finetuned.jld2", create_isolated_graph(g)),
    ("Random_Finetuned", "Resultados/test-7/checkpoints/params_random_finetuned.jld2", create_random_graph(g))
]

all_metrics = []
mkpath("Resultados/test-7/plots/finetuned_study")

for (name, param_path, g_topo) in cases
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

        for state_idx in [1, 2, 3, 4, 5, 8] # NY, NJ, CA, GA, IL, FL
            p_pred = plot(title="Fine-Tuned: $name - $(states[state_idx])", xlabel="Days", ylabel="Log Cases")
            scatter!(p_pred, tsteps_full, y_true[:, state_idx], label="Data", color=:black, markersize=1, alpha=0.3)
            plot!(p_pred, tsteps_full, pred_t[:, state_idx], label="Prediction", color=:blue, linewidth=2)
            vline!(p_pred, [train_end_idx], label="Train End (180)", color=:blue, linestyle=:dash)
            savefig(p_pred, "Resultados/test-7/plots/finetuned_study/pred_$(name)_$(states[state_idx]).png")
        end
    else
        println("Missing params for $name at $param_path")
    end
end

if !isempty(all_metrics)
    summary_df = DataFrame(all_metrics)
    println("\n=== SUMMARY OF FINE-TUNED RESULTS ===")
    println(summary_df)
    CSV.write("Resultados/test-7/study_results_finetuned.csv", summary_df)
end

println("Fine-Tuned Evaluation Complete.")
