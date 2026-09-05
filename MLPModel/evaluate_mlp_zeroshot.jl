# Import libraries
using Pkg
println("Current working directory: ", pwd())
Pkg.activate("..")

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, Random, Plots, StatsPlots, CubicSplines
using NPZ, JSON
import Plots: plot, savefig

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("Running evaluation on 12 States with ", Threads.nthreads(), " threads")

# Create save paths
errors_path = "./Plots/errors/mlp_zeroshot"
pred_path = "./Plots/dynamics/mlp_zeroshot"
mkdir(errors_path)
mkdir(pred_path)

# -------------------------------------------------------------
# 1. Data Loading 
# -------------------------------------------------------------

# Normal
train_states = [
    "AL", "CA", "CO", "CT", "DC", "DE", "FL", "IA", "ID", "IL", "IN", "KS", "KY",
    "LA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "NE", "NH", "NJ", "NM",
    "NY", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WI", "WV", "WY"
]

target_states = ["OH", "GA", "MA", "PA", "AR", "OK", "ND", "VT", "WA", "AZ", "UT", "NV"]


all_states = vcat(train_states, target_states)

# import adjacency matrix 
df_adj_full = CSV.read("../Data/fitted_matrix/flow_mat_exp_gravity_fitted.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
n_nodes = length(all_states)
name_to_index = Dict(name => i for (i, name) in enumerate(col_names))
indices = [name_to_index[s] for s in all_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

function controlled_normalize(A; alpha=0.6f0)
    A_no_diag = copy(A)
    # Remove self-loops entirely for the neighbor calculation
    A_no_diag[diagind(A_no_diag)] .= 0.0
    # Normalize ONLY the neighbor weights
    # matrix has flows origin state on rows, destination on columns, normalize arriving signals on columns to one
    col_sums = sum(A_no_diag, dims=1)
    A_neigh_norm = Float32.(A_no_diag ./ (col_sums .+ 1e-8))
    # Create an Identity matrix
    I_mat = Matrix{Float32}(I, size(A)...)
    # Blend: alpha * Self + (1 - alpha) * Neighbors
    return alpha .* I_mat .+ (1.0f0 - alpha) .* A_neigh_norm
end

A = controlled_normalize(adj_sub; alpha=0.5f0)

# load data
features_raw = NPZ.npzread("../Data/data_all.npz")  
# load calendar dates
dates = JSON.parsefile("../Data/dates.json")
# Make data tensor
n_vars, n_times = size(features_raw["NY"])
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(all_states)
    X_tensor_all[:, :, i] = features_raw[state]
end

# select variables
idx_list = [1,2,3,6] 
# Ensure variable 1 (target: daily infections) is always included
if !(1 in idx_list)
    push!(idx_list, 1)
end
idx_list = sort(unique(idx_list)) # Remove duplicates and keep things ordered
X_tensor = X_tensor_all[idx_list,:,:]

# shift variables
lags = [7,7,14]
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

n_vars, n_times, n_nodes = size(X_aligned)
println(n_vars, n_times, n_nodes)

# Normalize
X_norm = log.(X_aligned .+ 1.0f0)
# Time
tsteps = Float32.(collect(0:n_times-1))
# Define matrix of splines functions: Num_var x Num_nodes
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

println("Data loaded. Nodes: $n_nodes, Time: $n_times, Features: $n_vars")
println("Splines ready.")

# -------------------------------------------------------------
# 2. Model Definition 
# -------------------------------------------------------------

latent_dim = 3
nin_target = 1
nin_covar = size(covariate_splines,1)
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

# Frozen Dropout Layer (Unchanged)
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

# NEW LAYER (Unchanged)
struct WeightedGraphLayer{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    act::F
end

WeightedGraphLayer(in_dims::Int, out_dims::Int; act=identity) = 
    WeightedGraphLayer(in_dims, out_dims, act)

function Lux.initialparameters(rng::Random.AbstractRNG, l::WeightedGraphLayer)
    (W = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
     b = zeros(Float32, l.out_dims, 1))
end

Lux.initialstates(::Random.AbstractRNG, ::WeightedGraphLayer) = NamedTuple()

function (l::WeightedGraphLayer)(x::AbstractMatrix, A::AbstractMatrix, ps, st)
    x_agg = x * A                        
    out = l.act.(ps.W * x_agg .+ ps.b)   
    return out, st
end

# NEW ARCHITECTURE (Updated)
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
        Lux.Dense(nin, nhidden, tanh),                  # MLP Input layer
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden; act=tanh), # GNN layer 1
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden; act=tanh), # GNN layer 2
        FrozenDropout(drop_p),
        Lux.Dense(nhidden, nout)                        # MLP Output layer
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

# Forward pass
function (layer::ExplicitGNN)(A::AbstractMatrix, x::AbstractMatrix, ps, st)
    # Layer 1: MLP (Dense) - Does not use adjacency matrix A
    x, st_c1 = layer.layer1(x, ps.layer1, st.layer1)
    x, st_d1 = layer.drop1(x, ps.drop1, st.drop1)

    # Layer 2: GNN - Uses adjacency matrix A
    x, st_c2 = layer.layer2(x, A, ps.layer2, st.layer2)
    x, st_d2 = layer.drop2(x, ps.drop2, st.drop2)

    # Layer 3: GNN - Uses adjacency matrix A
    x, st_c3 = layer.layer3(x, A, ps.layer3, st.layer3)
    x, st_d3 = layer.drop3(x, ps.drop3, st.drop3)

    # Layer 4: MLP (Dense) - Does not use adjacency matrix A
    x, st_c4 = layer.layer4(x, ps.layer4, st.layer4)

    new_st = (layer1=st_c1, drop1=st_d1, layer2=st_c2, drop2=st_d2, 
              layer3=st_c3, drop3=st_d3, layer4=st_c4)
    return x, new_st
end

# Initialize Model (Width=64)

println("Initializing GNN (Width 64)...")
gnn = ExplicitGNN(nin_tot, 64, nout, 0.0) 
ps_gnn, st_gnn = Lux.setup(rng, gnn)
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) #|>f64
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features)

# -------------------------------------------------------------
# 3. Prediciton
# -------------------------------------------------------------

# Load optimized parameters from ensemble
println("Loading Trained Weights from Ensemble...")
ensemble_dir = "./Parameters/Par_zeroshot"

# Grab all jld2 files in the folder
ensemble_files = filter(f -> startswith(basename(f), "params_") && endswith(f, ".jld2"), readdir(ensemble_dir, join=true))
n_models = length(ensemble_files)
println("Found $n_models model parameter sets in $ensemble_dir")

# Load all parameter sets into a vector
ensemble_ps = []
for file in ensemble_files
    data = JLD2.load(file)
    param_key = haskey(data, "ps_final") ? "ps_final" : first(keys(data))
    push!(ensemble_ps, data[param_key])
end

function predict_full(model, ps, st, u0, tsteps, splines)
    # FrozenDropout code handles nothing if mask dropout is not defined
    # transfer learning for latent variables
    train_latents = ps.latent_features
    μ_lat = Float32(mean(train_latents))
    σ_lat = Float32(std(train_latents))
    
    # Random initialization matching training distribution
    new_latents_target = (randn(rng, Float32, size(train_latents,1), length(target_states)) .* σ_lat) .+ μ_lat
    new_latents = hcat(train_latents, new_latents_target)
    
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        col_t = map(s -> s(t), splines)
        model_input = vcat(u_reshaped, col_t, new_latents)
        y, _ = model(A, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps, saveat=tsteps, reltol=1f-3, abstol=1f-3)

    sol_mat = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

println("Predicting with Ensemble and Latent Uncertainty...")
u0 = X_norm[1, 1, :]

# Define how many latent samples to run PER model
n_latent_samples = 10 
n_total_samples = n_models * n_latent_samples
pred_samples = Array{Float32}(undef, n_total_samples, length(tsteps), n_nodes)

# Loop through both the ensemble parameters and the latent samples
sample_idx = 1
for m_idx in 1:n_models
    ps_current = ensemble_ps[m_idx]
    
    for s in 1:n_latent_samples
        pred = predict_full(gnn, ps_current, st_gnn, u0, tsteps, covariate_splines)
        pred_samples[sample_idx, :, :] .= pred[1, :, :]
        global sample_idx += 1
    end
end

# Compute predictions statistics 
# Mean
pred_mean = dropdims(median(pred_samples, dims=1), dims=1)
# 95% Confidence Interval (2.5% - 97.5%)
pred_lower_95 = dropdims(mapslices(x -> quantile(x, 0.025), pred_samples, dims=1), dims=1)
pred_upper_95 = dropdims(mapslices(x -> quantile(x, 0.975), pred_samples, dims=1), dims=1)
# 50% Confidence Interval (25% - 75%)
pred_lower_50 = dropdims(mapslices(x -> quantile(x, 0.25), pred_samples, dims=1), dims=1)
pred_upper_50 = dropdims(mapslices(x -> quantile(x, 0.75), pred_samples, dims=1), dims=1)

# Denormalize
pred_mean = exp.(pred_mean) .- 1
pred_lower_95 = exp.(pred_lower_95) .- 1
pred_upper_95 = exp.(pred_upper_95) .- 1
pred_lower_50 = exp.(pred_lower_50) .- 1
pred_upper_50 = exp.(pred_upper_50) .- 1

# -------------------------------------------------------------
# 4. Plot predicitons
# -------------------------------------------------------------

for i in eachindex(all_states)
    state_name = all_states[i]
    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    lower95 = pred_lower_95[:, i]
    upper95 = pred_upper_95[:, i]
    lower50 = pred_lower_50[:, i]
    upper50 = pred_upper_50[:, i]
    ribbon95 = (mean_curve .- lower95, upper95 .- mean_curve)
    ribbon50 = (mean_curve .- lower50, upper50 .- mean_curve)

    # Metrics
    mse = mean(abs2, data_curve .- mean_curve)
    mae = mean(abs, data_curve .- mean_curve)

    if state_name in target_states
        name = "$state_name *"
    else
        name = "$state_name"
    end

    p = plot(tsteps, data_curve, label="Data", color=:black, alpha=0.5, linewidth=1.5)
    plot!(p,tsteps, mean_curve, ribbon=ribbon95, color=:blue, fillalpha=0.2, linewidth=2, label=nothing)
    plot!(p,tsteps, mean_curve, ribbon=ribbon50, color=:blue, fillalpha=0.4, linewidth=2, label="Prediction")
    #plot!(p,tsteps, mean_curve, color=:blue, fillalpha=0.4, linewidth=2, label="Prediction")
    title!(p, "$name (Mean Test MAE: $(round(mae, digits=3)))")
    display(p)
    savefig(joinpath(pred_path, "dyn_state_$name.png"))
end

# Single plot for testing states

subplots = []
for i in eachindex(all_states)
    state_name = all_states[i]
    # Skip states that are not in our target list
    if !(state_name in target_states)
        continue
    end

    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    lower95 = pred_lower_95[:, i]
    upper95 = pred_upper_95[:, i]
    lower50 = pred_lower_50[:, i]
    upper50 = pred_upper_50[:, i]
    ribbon95 = (mean_curve .- lower95, upper95 .- mean_curve)
    ribbon50 = (mean_curve .- lower50, upper50 .- mean_curve)

    # Metrics
    mse = mean(abs2, data_curve .- mean_curve)
    mae = mean(abs, data_curve .- mean_curve)

    # Create the individual plot 
    p = plot(tsteps, data_curve, label="Data", color=:black, alpha=0.5, linewidth=1.5)
    plot!(p, tsteps, mean_curve, ribbon=ribbon95, color=:blue, fillalpha=0.2, linewidth=2, label=nothing)
    plot!(p,tsteps, mean_curve, ribbon=ribbon50, color=:blue, fillalpha=0.4, linewidth=2, label="Prediction")
    #plot!(p,tsteps, mean_curve, color=:blue, fillalpha=0.4, linewidth=2, label="Prediction")
    title!(p, "$state_name (Mean Test MAE: $(round(mae, digits=3)))")

    # Push the plot to our array instead of displaying it
    push!(subplots, p)
end


# Combine the 12 plots into a 4x3 grid

final_figure = plot(subplots..., layout=(4, 3), size=(1200, 800));
display(final_figure)
savefig("./Plots/testing/mlp_zeroshot/testing_states.png")

# -------------------------------------------------------------
# 5. Errors tables
# -------------------------------------------------------------

# Load state to cluster dictionary (from test9)
state_to_cluster = load("../Data/Dict_clusters/state_cluster_mapping_expFit.jld2", "state_to_cluster") 

# Load Population vector for per-capita scaling
pop_df = CSV.read("../Data/us_pop_by_state.csv", DataFrame)
pop_df = pop_df[:, ["state_code", "2020_census"]]
pop_df = pop_df[in.(pop_df.state_code, Ref(all_states)), :]
pop_df = pop_df[sortperm(indexin(pop_df.state_code, all_states)), :]
population = pop_df."2020_census"

# Helper function for Weighted Interval Score (WIS)
function compute_wis(y_true, lower_95, upper_95, lower_50, upper_50, median_pred)
    function interval_score(y, l, u, alpha)
        penalty_l = (2 / alpha) * (l - y) * (y < l)
        penalty_u = (2 / alpha) * (y - u) * (y > u)
        return (u - l) + penalty_l + penalty_u
    end

    score_95 = mean(interval_score.(y_true, lower_95, upper_95, 0.05))
    score_50 = mean(interval_score.(y_true, lower_50, upper_50, 0.50))
    ae_median = mean(abs.(y_true .- median_pred))

    # CDC Forecast Hub weighting: (0.5 * AE + 0.025 * IS_95 + 0.25 * IS_50) / 0.775
    return (0.5 * ae_median + 0.025 * score_95 + 0.25 * score_50) / (0.5 + 0.025 + 0.25)
end

n_samples, n_time, n_nodes = size(pred_samples)

# 1. Sample-level evaluation matrices (calculates metrics over ALL ensemble iterations)
mae_norm_samples   = zeros(n_samples, n_nodes)
wmape_samples      = zeros(n_samples, n_nodes)
per_capita_samples = zeros(n_samples, n_nodes)
peak_amp_samples   = zeros(n_samples, n_nodes)
peak_shift_samples = zeros(Int, n_samples, n_nodes)

for s in 1:n_samples
    for i in 1:n_nodes
        # Normalized scale (training space)
        p_norm = pred_samples[s, :, i]
        t_norm = X_norm[1, :, i]
        mae_norm_samples[s, i] = mean(abs.(p_norm .- t_norm))

        # Denormalized scale (real case counts)
        p_true = exp.(p_norm) .- 1.0f0
        t_true = exp.(t_norm) .- 1.0f0

        # Weighted Absolute Percentage Error (wMAPE)
        wmape_samples[s, i] = sum(abs.(p_true .- t_true)) / (sum(t_true) + 1e-8)

        # Population-normalized MAE (Daily cases per 100k people)
        mae_denorm = mean(abs.(p_true .- t_true))
        per_capita_samples[s, i] = (mae_denorm / population[i]) * 100_000

        # Peak dynamics
        max_p = maximum(p_true)
        max_t = maximum(t_true)
        peak_amp_samples[s, i] = (max_p - max_t) / (max_t + 1e-8)
        peak_shift_samples[s, i] = argmax(p_true) - argmax(t_true)
    end
end

# 2. Distributional metrics (evaluated across ensemble boundaries)
wis_scores  = zeros(n_nodes)
coverage_95 = zeros(n_nodes)
coverage_50 = zeros(n_nodes)

for i in 1:n_nodes
    truth = exp.(X_norm[1, :, i]) .- 1.0f0

    # Empirical coverage calibration checks
    coverage_95[i] = mean((pred_lower_95[:, i] .<= truth) .& (truth .<= pred_upper_95[:, i]))
    coverage_50[i] = mean((pred_lower_50[:, i] .<= truth) .& (truth .<= pred_upper_50[:, i]))

    # Weighted Interval Score
    wis_scores[i] = compute_wis(
        truth, 
        pred_lower_95[:, i], pred_upper_95[:, i],
        pred_lower_50[:, i], pred_upper_50[:, i],
        pred_mean[:, i]
    )
end

# 3. Construct master DataFrame with 95% CIs
summary_df = DataFrame(
    "State" => all_states,
    "Training_split" => [s in train_states ? "Train" : "Test" for s in all_states],
    "Cluster" => [get(state_to_cluster, s, "Unknown") for s in all_states],
    
    # Interval Calibration & Uncertainty Quality
    "Coverage_95" => round.(coverage_95, digits=2),
    "Coverage_50" => round.(coverage_50, digits=2),
    "WIS" => round.(wis_scores, digits=3),
    
    # Loss Space Metric: Log-Normalized MAE (Mean & 95% CI)
    "MAE_norm_mean" => round.(vec(mean(mae_norm_samples, dims=1)), digits=4),
    "MAE_norm_q025" => round.([quantile(col, 0.025) for col in eachcol(mae_norm_samples)], digits=4),
    "MAE_norm_q975" => round.([quantile(col, 0.975) for col in eachcol(mae_norm_samples)], digits=4),

    # Denormalized Metrics: wMAPE (Mean & 95% CI)
    "wMAPE_mean" => round.(vec(mean(wmape_samples, dims=1)), digits=3),
    "wMAPE_q025" => round.([quantile(col, 0.025) for col in eachcol(wmape_samples)], digits=3),
    "wMAPE_q975" => round.([quantile(col, 0.975) for col in eachcol(wmape_samples)], digits=3),

    # Denormalized Metrics: Per Capita Error per 100k (Mean & 95% CI)
    "PerCapita100k_mean" => round.(vec(mean(per_capita_samples, dims=1)), digits=2),
    "PerCapita100k_q025" => round.([quantile(col, 0.025) for col in eachcol(per_capita_samples)], digits=2),
    "PerCapita100k_q975" => round.([quantile(col, 0.975) for col in eachcol(per_capita_samples)], digits=2),

    # Wave Dynamics: Peak Shift in Days & Amplitude Ratio (Mean & 95% CI)
    "PeakShift_mean" => round.(vec(mean(peak_shift_samples, dims=1)), digits=1),
    "PeakShift_q025" => [quantile(col, 0.025) for col in eachcol(peak_shift_samples)],
    "PeakShift_q975" => [quantile(col, 0.975) for col in eachcol(peak_shift_samples)],
    
    "PeakAmp_ratio_mean" => round.(vec(mean(peak_amp_samples, dims=1)), digits=3),
    "PeakAmp_ratio_q025" => round.([quantile(col, 0.025) for col in eachcol(peak_amp_samples)], digits=3),
    "PeakAmp_ratio_q975" => round.([quantile(col, 0.975) for col in eachcol(peak_amp_samples)], digits=3)
)

# Export Summary CSV
CSV.write("./Tables/errors_summary_zeroshot.csv", summary_df)

# 4. Print Aggregated Split Statistics
for split in ["Train", "Test"]
    sub = filter(:Training_split => ==(split), summary_df)
    println("==========================================")
    println("           $split SPLIT SUMMARY           ")
    println("==========================================")
    println("Coverage (95% CI):   ", round(mean(sub.Coverage_95), digits=2))
    println("Coverage (50% CI):   ", round(mean(sub.Coverage_50), digits=2))
    println("Mean WIS:            ", round(mean(sub.WIS), digits=3))
    println("Mean Norm MAE:       ", round(mean(sub.MAE_norm_mean), digits=4))
    println("Mean wMAPE:          ", round(mean(sub.wMAPE_mean), digits=3))
    println("Mean Per-Capita MAE: ", round(mean(sub.PerCapita100k_mean), digits=2), " cases / 100k")
    println("Mean Peak Shift:     ", round(mean(sub.PeakShift_mean), digits=1), " days")
    println("Mean Peak Amp Error: ", round(mean(sub.PeakAmp_ratio_mean) * 100, digits=1), "%")
    println()
end

# -------------------------------------------------------------
# 6. Errors plots
# -------------------------------------------------------------

# Boxplot training vs test errors
boxplot(summary_df.Training_split, summary_df.wMAPE_mean, xlabel = "Dataset split", ylabel = "wMAPE", title = "Distribution of prediction errors", legend = false);
scatter!(summary_df.Training_split, summary_df.wMAPE_mean, color=:black, alpha=0.7)

savefig(joinpath(errors_path,"errors_ens10.png"))

# Boxplot clusters
train_df = summary_df[summary_df.Training_split .== "Train", :]
test_df  = summary_df[summary_df.Training_split .== "Test", :]

p_train = boxplot(train_df.Cluster, train_df.wMAPE_mean, xlabel = "Cluster", ylabel = "wMAPE", title = "Training states: error distribution by cluster", legend = false);
scatter!(p_train, train_df.Cluster, train_df.wMAPE_mean, color = :black, alpha = 0.7);
display(p_train)

savefig(joinpath(errors_path,"train_cluster_ens10.png"))

p_test = boxplot(test_df.Cluster, test_df.wMAPE_mean, xlabel = "Cluster", ylabel = "wMAPE", title = "Testing states: error distribution by cluster", legend = false);
scatter!(p_test, test_df.Cluster, test_df.wMAPE_mean, color = :black, alpha = 0.7);
display(p_test)

savefig(joinpath(errors_path,"test_cluster_ens10.png"))

# Sorted MAE plot colored by Train/Test
sorted_df = sort(summary_df, :wMAPE_mean) # sort states by error
colors = [s == "Train" ? :blue : :red for s in sorted_df.Training_split] # colors

p = scatter(1:nrow(sorted_df), sorted_df.wMAPE_mean, color = colors, xlabel = "States (sorted by wMAPE)", ylabel = "wMAPE", title = "Prediction error across states", legend = false);
xticks!(1:nrow(sorted_df), sorted_df.State, rotation = 90);
display(p)

savefig(joinpath(errors_path,"sortedMape_ens10.png"))

# Heatmap: errors over USA map

using PlotlyJS

# wMAPE
# differentiate state name color if it's test or train state
text_colors = [state in target_states ? "blue" : "black" for state in summary_df.State]

# Per capita error
trace_color = choropleth(
    locations = summary_df.State, # Make sure this points to the ABBREVIATIONS
    locationmode = "USA-states",
    z = summary_df.PerCapita100k_mean,      # The color value
    colorscale = "Reds",           # "Reds", "Viridis", or "YlOrRd" work well for errors
    marker_line_color = "white",   # White borders between states look clean
    colorbar_title = "Error per 100k"
)

trace_labels = scattergeo(
    locations = summary_df.State,
    locationmode = "USA-states",
    text = summary_df.State,     # This dictates what text appears
    mode = "text",                # Forces it to show text instead of dots
    textfont = attr(
        color = text_colors,          # Dark text usually contrasts best against the "Reds" scale
        size = 7,
        weight = "bold"
    ),
    showlegend = false,
    hoverinfo = "skip"            # Prevents hover tooltips from popping up over the text
)

# Define the layout and scope
layout = Layout(
    title = "Error per 100k across USA",
    geo = attr(
        scope = "usa",
        projection_type = "albers usa",
        showlakes = false
    ),
    margin = attr(l=0, r=0, t=40, b=0)
)

p_map = PlotlyJS.plot(
    [trace_color, trace_labels], 
    layout, 
    config = PlotConfig(staticPlot = true) 
)
display(p_map)

PlotlyJS.savefig(p_map, joinpath(errors_path,"heatmap_capita_ens10.png"), width=1000, height=550)

