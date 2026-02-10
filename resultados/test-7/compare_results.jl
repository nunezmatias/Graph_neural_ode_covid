# Test 7: Compare Results Across Topologies
# Generates comparison plots and summary statistics

using CSV, DataFrames, JLD2, Plots, Statistics

println("=== Test 7: Results Comparison ===\n")

# Load loss histories
loss_full = CSV.read("Resultados/test-7/checkpoints/loss_full.csv", DataFrame)
loss_isolated = CSV.read("Resultados/test-7/checkpoints/loss_isolated.csv", DataFrame)
loss_random = CSV.read("Resultados/test-7/checkpoints/loss_random.csv", DataFrame)

println("Loaded loss histories:")
println("  Full: ", nrow(loss_full), " epochs")
println("  Isolated: ", nrow(loss_isolated), " epochs")
println("  Random: ", nrow(loss_random), " epochs\n")

# Summary statistics
function summarize_loss(df, name)
    final_loss = df.loss[end]
    min_loss = minimum(df.loss)
    mean_last_100 = mean(df.loss[max(1, end - 99):end])

    println("$name:")
    println("  Final Loss: $(round(final_loss, digits=6))")
    println("  Min Loss: $(round(min_loss, digits=6))")
    println("  Mean (last 100): $(round(mean_last_100, digits=6))")

    return (final=final_loss, min=min_loss, mean_last=mean_last_100)
end

stats_full = summarize_loss(loss_full, "Full Graph")
stats_iso = summarize_loss(loss_isolated, "Isolated Nodes")
stats_rand = summarize_loss(loss_random, "Random Graph")

# Calculate relative differences
println("\nRelative Performance (vs Full Graph):")
diff_iso = ((stats_iso.final - stats_full.final) / stats_full.final) * 100
diff_rand = ((stats_rand.final - stats_full.final) / stats_full.final) * 100

println("  Isolated: $(round(diff_iso, digits=2))% $(diff_iso > 0 ? "worse" : "better")")
println("  Random: $(round(diff_rand, digits=2))% $(diff_rand > 0 ? "worse" : "better")")

# Hypothesis test
println("\n" * "="^60)
if abs(diff_iso) < 10 && abs(diff_rand) < 10
    println("RESULT: Primary Hypothesis CONFIRMED")
    println("Graph structure contributes <10% to performance.")
    println("Model is dominated by local inputs and latent features.")
elseif abs(diff_iso) > 50
    println("RESULT: Null Hypothesis REJECTED")
    println("Graph structure is ESSENTIAL (>50% performance drop).")
    println("Spatial diffusion is critical to model dynamics.")
else
    println("RESULT: Moderate Graph Contribution")
    println("Graph helps (10-50% improvement), but not critical.")
end
println("="^60)

# Plot 1: Loss Curves
p1 = plot(
    title="Training Loss Comparison",
    xlabel="Epoch",
    ylabel="MSE Loss (log-space)",
    legend=:topright,
    size=(1000, 600),
    dpi=150,
    yscale=:log10
)

plot!(p1, loss_full.epoch, loss_full.loss, label="Full Graph", lw=2, color=:blue)
plot!(p1, loss_isolated.epoch, loss_isolated.loss, label="Isolated Nodes", lw=2, color=:green, linestyle=:dash)
plot!(p1, loss_random.epoch, loss_random.loss, label="Random Graph", lw=2, color=:red, linestyle=:dot)

savefig(p1, "Resultados/test-7/plots/loss_comparison.png")
println("\nSaved: plots/loss_comparison.png")

# Plot 2: Final Loss Bar Chart
p2 = bar(
    ["Full", "Isolated", "Random"],
    [stats_full.final, stats_iso.final, stats_rand.final],
    title="Final Loss Comparison",
    ylabel="MSE Loss",
    legend=false,
    color=[:blue, :green, :red],
    size=(600, 500),
    dpi=150
)

savefig(p2, "Resultados/test-7/plots/final_loss_bar.png")
println("Saved: plots/final_loss_bar.png")

# Save summary table
summary_df = DataFrame(
    Topology=["Full", "Isolated", "Random"],
    FinalLoss=[stats_full.final, stats_iso.final, stats_rand.final],
    MinLoss=[stats_full.min, stats_iso.min, stats_rand.min],
    MeanLast100=[stats_full.mean_last, stats_iso.mean_last, stats_rand.mean_last],
    RelativeDiff=[0.0, diff_iso, diff_rand]
)

CSV.write("Resultados/test-7/results_summary.csv", summary_df)
println("Saved: results_summary.csv")

println("\nAnalysis complete!")
