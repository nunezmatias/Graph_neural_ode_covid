# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, Random, Plots, StatsPlots, CubicSplines
using NPZ, JSON

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("Running evaluation on 9 States with ", Threads.nthreads(), " threads")

# -------------------------------------------------------------
# 1. Data Loading 
# -------------------------------------------------------------

# leave-out states for testing
target_states = ["AZ", "LA", "MA", "MD", "NM", "NV", "RI", "TN", "UT"]

train_states = [
    "AL", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NY", "OH", "OK", "OR", "PA", "SC", "SD", "TX", "VA", "VT", "WA", "WI", "WV", "WY"
]

# import adjacency matrix 
df_adj_full = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states = names(df_adj_full)[2:end]
#indices = [findfirst(x -> x == s, all_states) for s in train_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
#adj_sub = adj_raw[indices, indices]
adj_raw[adj_raw.<0.05] .= 0.0
adj_norm = adj_raw ./ maximum(adj_raw)
n_nodes = length(all_states)
for i in 1:n_nodes
    adj_norm[i, i] = 0.0
end
# Make GNN graph
g = GNNGraph(sparse(adj_norm + I))
# load data
features_raw = NPZ.npzread("Data/data_all.npz")  

# Make data tensor
n_vars, n_times = size(features_raw["NY"])
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(all_states)
    X_tensor_all[:, :, i] = features_raw[state]
end

# select variables
idx_list = [6] 
# Ensure variable 1 (target: daily infections) is always included
if !(1 in idx_list)
    push!(idx_list, 1)
end
idx_list = sort(unique(idx_list)) # Remove duplicates and keep things ordered
X_tensor = X_tensor_all[idx_list,:,:]

# shift variables
lags = [14]
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

# MODIFIED: Hidden Dim = 64
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
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x, st_d1 = (x, st_d1) # fix tuple unpacking if dropout returns tuple

    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)

    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)

    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
end

function sample_dropout_masks(model, st, x_shape)
    d1, d2, d3 = model.drop1, model.drop2, model.drop3
    # Use 64 as hidden dim
    mask1 = rand(st.drop1.rng, Float32, (64, x_shape[2])) .> d1.p
    mask2 = rand(st.drop2.rng, Float32, (64, x_shape[2])) .> d2.p
    mask3 = rand(st.drop3.rng, Float32, (64, x_shape[2])) .> d3.p

    st_d1 = (mask=mask1 ./ (1 - d1.p), rng=st.drop1.rng)
    st_d2 = (mask=mask2 ./ (1 - d2.p), rng=st.drop2.rng)
    st_d3 = (mask=mask3 ./ (1 - d3.p), rng=st.drop3.rng)

    return (
        layer1=st.layer1, drop1=st_d1,
        layer2=st.layer2, drop2=st_d2,
        layer3=st.layer3, drop3=st_d3,
        layer4=st.layer4
    )
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

# Load optimized parameters
println("Loading Trained Weights...")
@load "checkpoints_shift/params_shift_idx1_6.jld2" ps_final

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
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps, saveat=tsteps, reltol=1f-3, abstol=1f-3) # 1f-5 instead of 1e-5 for float32

    sol_mat = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

println("Predicting...")
# Make MC predictions, stochasticity given by latent variables
u0 = X_norm[1, 1, :]
n_samples = 100
pred_samples = Array{Float32}(undef, n_samples, length(tsteps), n_nodes)
for s in 1:n_samples
    pred = predict_full(gnn, ps_final, st_gnn, u0, tsteps, covariate_splines)
    pred_samples[s, :, :] .= pred[1, :, :]
end

# Compute predictions statistics
pred_mean = dropdims(mean(pred_samples, dims=1), dims=1)
pred_q05 = mapslices(x -> quantile(x, 0.05), pred_samples; dims=1) |> x -> dropdims(x, dims=1)
pred_q95 = mapslices(x -> quantile(x, 0.95), pred_samples; dims=1) |> x -> dropdims(x, dims=1)
# Denormalize
pred_mean = exp.(pred_mean) .- 1
pred_q05 = exp.(pred_q05) .- 1
pred_q95 = exp.(pred_q95) .- 1

# -------------------------------------------------------------
# 4. Plot predicitons
# -------------------------------------------------------------

for i in eachindex(all_states)

    state_name = all_states[i]

    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    lower = pred_q05[:, i]
    upper = pred_q95[:, i]
    ribbon = (mean_curve .- lower, upper .- mean_curve)

    # Metrics
    mse = mean(abs2, data_curve .- mean_curve)
    mae = mean(abs, data_curve .- mean_curve)

    if state_name in target_states
        name = "$state_name *"
    else
        name = "$state_name"
    end

    p = plot(tsteps, data_curve, label="Data", color=:black, alpha=0.5, linewidth=1.5)
    plot!(p,tsteps, mean_curve, ribbon=ribbon, color=:blue, fillalpha=0.25, linewidth=2, label="Prediction")
    title!(p, "$name (Mean Test MAE: $(round(mae, digits=3)))")
    display(p)
    savefig("Plots/dynamics/shift1/dyn_state_$name.png")
end


# Single plot for testing states
subplots = []

for i in eachindex(all_states)
    state_name = all_states[i]

    # Skip states that are not in target list
    if !(state_name in target_states)
        continue
    end

    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    lower = pred_q05[:, i]
    upper = pred_q95[:, i]
    ribbon = (mean_curve .- lower, upper .- mean_curve)

    # Metrics
    mse = mean(abs2, data_curve .- mean_curve)
    mae = mean(abs, data_curve .- mean_curve)

    # Create the individual plot
    p = plot(tsteps, data_curve, label="Data", color=:black, alpha=0.5, linewidth=1.5)
    plot!(p, tsteps, mean_curve, ribbon=ribbon, color=:blue, fillalpha=0.25, linewidth=2, label="Prediction")
    title!(p, "$state_name (Mean Test MAE: $(round(mae, digits=3)))")
    
    # Push the plot to array
    push!(subplots, p)
end

# Combine the 9 plots into a 3x3 grid
final_figure = plot(subplots..., layout=(3, 3), size=(1200, 800));
display(final_figure)

savefig("Plots/testing/testing_shift1.png")

# -------------------------------------------------------------
# 5. Errors tables
# -------------------------------------------------------------

# Load state to cluster dictionary (from test9)
state_to_cluster = load("state_cluster_mapping.jld2", "state_to_cluster")

# Population vector
pop_df = CSV.read("us_pop_by_state.csv", DataFrame)
pop_df = pop_df[:, ["state_code", "2020_census"]]
pop_df = pop_df[in.(pop_df.state_code, Ref(all_states)), :]
pop_df = pop_df[sortperm(indexin(pop_df.state_code, all_states)), :]
population = pop_df."2020_census"

# Compute  statistics for each sample
n_samples, n_time, n_nodes = size(pred_samples)
mae_samples = zeros(n_samples, n_nodes)
wmape_samples = zeros(n_samples, n_nodes)
per_capita_samples = zeros(n_samples, n_nodes)
peak_shift_samples = zeros(Int, n_samples, n_nodes)

for s in 1:n_samples
    for i in 1:n_nodes
        # log scale
        p = pred_samples[s, :, i]
        t = X_norm[1, :, i]
        # real scale
        p_true = exp.(pred_samples[s, :, i]) .- 1
        t_true = exp.(X_norm[1, :, i]) .- 1

        abs_error = mean(abs.(p_true .- t_true))
        mae_samples[s, i] = abs_error
        wmape_samples[s, i] = sum(abs.(p_true .- t_true)) / (sum(t_true) + 1e-8)
        per_capita_samples[s, i] = (abs_error / population[i]) * 100_000

        pred_peak_day = argmax(p_true)
        true_peak_day = argmax(t_true)
        peak_shift_samples[s, i] = pred_peak_day - true_peak_day
    end
end

# compute summary statistics
mae_mean = mean(mae_samples, dims=1) |> vec
mae_q05 = [quantile(col, 0.05) for col in eachcol(mae_samples)]
mae_q95 = [quantile(col, 0.95) for col in eachcol(mae_samples)]

wmape_mean = mean(wmape_samples, dims=1) |> vec
wmape_q05 = [quantile(col, 0.05) for col in eachcol(wmape_samples)]
wmape_q95 = [quantile(col, 0.95) for col in eachcol(wmape_samples)]

per_capita_mean = mean(per_capita_samples, dims=1) |> vec
per_capita_q05 = [quantile(col, 0.05) for col in eachcol(per_capita_samples)]
per_capita_q95 = [quantile(col, 0.95) for col in eachcol(per_capita_samples)]

peak_shift_mean = vec(mean(peak_shift_samples, dims=1))
peak_shift_q05 = [quantile(col, 0.05) for col in eachcol(peak_shift_samples)]
peak_shift_q95 = [quantile(col, 0.95) for col in eachcol(peak_shift_samples)]

# Coverage
coverage = zeros(n_nodes)
for i in 1:n_nodes
    truth = exp.(X_norm[1, :, i]) .- 1
    lower = pred_q05[:, i]
    upper = pred_q95[:, i]
    coverage[i] = mean((truth .>= lower) .& (truth .<= upper))
end

# Peak timing (within +- 3 days) probability
peak_accuracy = zeros(n_nodes)
for i in 1:n_nodes
    peak_accuracy[i] =
        mean(abs.(peak_shift_samples[:, i]) .<= 3)
end

# Dataframe
summary_df = DataFrame(
    "State" => all_states,
    "Cluster" => [get(state_to_cluster, s, "Unknown") for s in all_states],
    "Training_split" => [s in train_states ? "Train" : "Test" for s in all_states],
    "Coverage" => round.(coverage, digits=2),
    "Peak timing" => peak_accuracy,

    "MAE_mean" => round.(mae_mean, digits=3),
    "MAE_q05" => round.(mae_q05, digits=3),
    "MAE_q95" => round.(mae_q95, digits=3),

    "wMAPE_mean" => round.(wmape_mean, digits=3),
    "wMAPE_q05" => round.(wmape_q05, digits=3),
    "wMAPE_q95" => round.(wmape_q95, digits=3),

    "per_capita_mean" => round.(per_capita_mean, digits=3),
    "per_capita_q05" => round.(per_capita_q05, digits=3),
    "per_capita_q95" => round.(per_capita_q95, digits=3),

    "PeakShift_mean" => round.(peak_shift_mean, digits=1),
    "PeakShift_q05" => round.(peak_shift_q05, digits=1),
    "PeakShift_q95" => round.(peak_shift_q95, digits=1)
)


# Save tables
CSV.write("Tables/errors_summary_shift1.csv", summary_df)

# -------------------------------------------------------------
# 6. Errors plots
# -------------------------------------------------------------

errors_path = "Plots/errors/shift1/"

# Boxplot training vs test errors
boxplot(summary_df.Training_split, summary_df.wMAPE_mean, xlabel = "Dataset split", ylabel = "wMAPE", title = "Distribution of prediction errors", legend = false);
scatter!(summary_df.Training_split, summary_df.wMAPE_mean, color=:black, alpha=0.7)

savefig(joinpath(errors_path,"errors_shift1.png"))

# Boxplot clusters
train_df = summary_df[summary_df.Training_split .== "Train", :]
test_df  = summary_df[summary_df.Training_split .== "Test", :]

p_train = boxplot(train_df.Cluster, train_df.wMAPE_mean, xlabel = "Cluster", ylabel = "wMAPE", title = "Training states: error distribution by cluster", legend = false);
scatter!(p_train, train_df.Cluster, train_df.wMAPE_mean, color = :black, alpha = 0.7);
display(p_train)

savefig(joinpath(errors_path,"train_cluster_shift1.png"))

p_test = boxplot(test_df.Cluster, test_df.wMAPE_mean, xlabel = "Cluster", ylabel = "wMAPE", title = "Testing states: error distribution by cluster", legend = false);
scatter!(p_test, test_df.Cluster, test_df.wMAPE_mean, color = :black, alpha = 0.7);
display(p_test)

savefig(joinpath(errors_path,"test_cluster_shift1.png"))

# Sorted MAE plot colored by Train/Test
sorted_df = sort(summary_df, :wMAPE_mean) # sort states by error
colors = [s == "Train" ? :blue : :red for s in sorted_df.Training_split] # colors

p = scatter(1:nrow(sorted_df), sorted_df.wMAPE_mean, color = colors, xlabel = "States (sorted by wMAPE)", ylabel = "wMAPE", title = "Prediction error across states", legend = false);
xticks!(1:nrow(sorted_df), sorted_df.State, rotation = 90);
display(p)

savefig(joinpath(errors_path,"sortedMape_shift1.png"))

# Heatmap: errors over USA map

using PlotlyJS

# wMAPE
# Create the geographic map (Choropleth)
trace_color = choropleth(
    locations = summary_df.State, 
    locationmode = "USA-states",
    z = summary_df.wMAPE_mean,      
    colorscale = "Reds",           
    marker_line_color = "white",   
    colorbar_title = "wMAPE"
)

trace_labels = scattergeo(
    locations = summary_df.State,
    locationmode = "USA-states",
    text = summary_df.State,   
    mode = "text",                # Forces it to show text instead of dots
    textfont = attr(
        color = "black",          
        size = 7,
        weight = "bold"
    ),
    showlegend = false,
    hoverinfo = "skip"            # Prevents hover tooltips from popping up over the text
)

# Define the layout and scope
layout = Layout(
    title = "Prediction Error (wMAPE) across USA",
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

PlotlyJS.savefig(p_map, "Plots/heatmaps/heatmap_wMAPE_shift1.png", width=1000, height=550)


# Per capita error
trace_color = choropleth(
    locations = summary_df.State, 
    locationmode = "USA-states",
    z = summary_df.per_capita_mean,      
    colorscale = "Reds",          
    marker_line_color = "white",   
    colorbar_title = "Error per 100k"
)

trace_labels = scattergeo(
    locations = summary_df.State,
    locationmode = "USA-states",
    text = summary_df.State,    
    mode = "text",              
    textfont = attr(
        color = "black",          
        size = 7,
        weight = "bold"
    ),
    showlegend = false,
    hoverinfo = "skip"            
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

PlotlyJS.savefig(p_map, "Plots/heatmaps/heatmap_capita_shift1.png", width=1000, height=550)











