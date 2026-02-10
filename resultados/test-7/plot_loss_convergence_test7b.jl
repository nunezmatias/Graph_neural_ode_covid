using Plots, DataFrames, CSV

function parse_loss_from_log(logfile)
    stages = Int[]
    epochs = Int[]
    losses = Float64[]
    current_offset = 0
    if !isfile(logfile)
        return DataFrame(Epoch=Int[], Loss=Float64[])
    end

    # Simple parser assuming strictly sequential logs from the last run
    for line in eachline(logfile)
        # Match pattern: "Fine-Tune Stage | Epoch 653 | Loss = 0.0685..."
        m = match(r"Fine-Tune Stage \| Epoch (\d+) \| Loss = ([\d\.]+)", line)
        if m !== nothing
            ep = parse(Int, m.captures[1])
            loss = parse(Float64, m.captures[2])
            push!(epochs, ep)
            push!(losses, loss)
        end
    end
    return DataFrame(Epoch=epochs, Loss=losses)
end

logs = [
    ("Full Graph", "Resultados/test-7/fine_tune_full.log", :red),
    ("Isolated", "Resultados/test-7/fine_tune_isolated.log", :green),
    ("Random", "Resultados/test-7/fine_tune_random.log", :blue)
]

p = plot(title="Training Loss Convergence (Deep Fine-Tuning)",
    xlabel="Epochs", ylabel="MSE Loss (Log Scale)",
    yaxis=:log, legend=:topright, size=(800, 500), thickness_scaling=1.2)

min_loss_val = Inf

for (name, path, color) in logs
    df = parse_loss_from_log(path)
    if !isempty(df)
        plot!(p, df.Epoch, df.Loss, label=name, color=color, linewidth=2, alpha=0.9)
        min_loss = minimum(df.Loss)
        if min_loss < min_loss_val
            global min_loss_val = min_loss
        end
        println("$name: Min Loss = $min_loss (latest epoch $(maximum(df.Epoch)))")
    else
        println("Warning: Log empty for $name")
    end
end

mkpath("Resultados/test-7/plots/convergence")
savefig(p, "Resultados/test-7/plots/convergence/loss_convergence_epoch650.png")
println("Plot saved to Hasilados/test-7/plots/convergence/loss_convergence_epoch650.png")
