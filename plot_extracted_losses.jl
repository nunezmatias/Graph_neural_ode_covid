using Pkg
Pkg.activate(".")
using Plots

log_files = [
    "Resultados/test-8BIS/training_phase1.log",
    "Resultados/test-8BIS/training_phase2.log",
    "Resultados/test-8BIS/training_phase3.log",
    "Resultados/test-8BIS/training_phase4.log",
    "Resultados/test-8BIS/training_phase5.log"
]

losses = Float64[]

for f in log_files
    if isfile(f)
        for line in eachline(f)
            if occursin("Loss:", line)
                try
                    parts = split(line, "Loss: ")
                    val = parse(Float64, split(parts[2], " ")[1])
                    if val <= 1000.0 # filter spikes
                        push!(losses, val)
                    end
                catch
                end
            end
        end
    end
end

if length(losses) > 0
    p = plot(losses, yaxis=:log10, label="Training Loss",
             xlabel="Logged Iterations", ylabel="Loss (Log10)",
             title="Test 8BIS: 5-Phase Continuous Loss",
             linewidth=2, color=:purple, size=(800, 500), margin=5Plots.mm)
             
    savefig(p, "Resultados/test-8BIS/plots/final_total_loss_curve.png")
    println("Successfully ploted $(length(losses)) loss points to Resultados/test-8BIS/plots/final_total_loss_curve.png")
else
    println("No losses found in the expected text logs. Falling back to stdout extraction.")
end
