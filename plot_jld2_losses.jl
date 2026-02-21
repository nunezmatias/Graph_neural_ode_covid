using Pkg
Pkg.activate(".")
using Plots
using JLD2

checkpoints = [
    "Resultados/test-8BIS/checkpoints/params_40s_fast.jld2",
    "Resultados/test-8BIS/checkpoints/params_40s_phase2.jld2",
    "Resultados/test-8BIS/checkpoints/params_40s_phase3.jld2",
    "Resultados/test-8BIS/checkpoints/params_40s_phase4_ext.jld2",
    "Resultados/test-8BIS/checkpoints/params_40s_phase5_final.jld2"
]

all_losses = Float64[]
phase_starts = Int[]

for ckpt in checkpoints
    if isfile(ckpt)
        try
            hist = JLD2.load(ckpt, "loss_history")
            push!(phase_starts, length(all_losses) + 1)
            append!(all_losses, hist)
        catch
            println("No loss_history found in ", ckpt)
        end
    else
        println("Checkpoint missing: ", ckpt)
    end
end

if length(all_losses) > 0
    # Filter explosive gradients just for the plot
    q99 = 1000.0
    filtered_losses = filter(x -> x <= q99, all_losses)
    
    p = plot(filtered_losses, yaxis=:log10, label="Training Loss",
             xlabel="Epochs", ylabel="Loss (Log10)",
             title="Test 8BIS: 5-Phase Curriculum Loss",
             linewidth=2, color=:purple, size=(800, 500), margin=5Plots.mm)
             
    for start_idx in phase_starts[2:end]
        vline!([start_idx], color=:black, ls=:dash, label="")
    end

    savefig(p, "Resultados/test-8BIS/plots/final_total_loss_curve.png")
    println("Successfully ploted $(length(filtered_losses)) cumulative epochs to Resultados/test-8BIS/plots/final_total_loss_curve.png")
else
    println("Error: Could not extract loss_history from any JLD2 files.")
end
