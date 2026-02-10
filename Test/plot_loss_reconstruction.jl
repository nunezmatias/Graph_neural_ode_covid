using Plots

# Reconstructed Loss Data (approximate from logs)
stages = [10, 20, 40, 60, 90, 120, 180]
final_loss = [0.23, 6.5, 36.3, 54.0, 56.5, 78.3, 62600.0]

# Create the plot
p = plot(stages, final_loss,
    marker=:circle,
    title="Test 4: Loss vs Curriculum Stage (No Covariates)",
    xlabel="Curriculum Window Size (Days)",
    ylabel="Final MSE Loss (Log Scale)",
    yscale=:log10,
    label="Test 4 (No Covars)",
    lw=3, color=:red, legend=:topleft
)

# Annotate the collapse
annotate!(180, 62600, text("  Collapse (>60k)", :bottom, :red, 8))
annotate!(120, 78.3, text("  78.3", :bottom, :red, 8))
annotate!(10, 0.23, text("  0.23", :bottom, :green, 8))

savefig(p, "Resultados/test-4/plots/loss_divergence.png")
println("Loss divergence plot saved.")
