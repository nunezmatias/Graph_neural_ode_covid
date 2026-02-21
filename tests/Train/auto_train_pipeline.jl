# auto_train_pipeline.jl

function parse_commandline()
    args = Dict{String,Any}(
        "smoke-test" => false,
        "full-run" => false,
        "phase1-ep" => 0, "phase2-ep" => 0, "phase3-ep" => 0,
        "phase4-ep" => 0, "phase5-ep" => 0, "phase6-ep" => 0
    )

    for (i, arg) in enumerate(ARGS)
        if arg == "--smoke-test"
            args["smoke-test"] = true
        elseif arg == "--full-run"
            args["full-run"] = true
        elseif startswith(arg, "--phase") && contains(arg, "-ep=")
            # Example: --phase1-ep=10
            parts = split(arg, "=")
            key = replace(parts[1], "--" => "")
            if haskey(args, key)
                args[key] = parse(Int, parts[2])
            end
        end
    end
    return args
end

function get_target_epochs(args)
    if args["smoke-test"]
        return [2, 2, 2, 2, 2, 2]
    elseif args["full-run"]
        # Historic epoch counts used for Test-8BIS
        return [60, 50, 60, 70, 150, 150]
    else
        ep1 = args["phase1-ep"] > 0 ? args["phase1-ep"] : 10
        ep2 = args["phase2-ep"] > 0 ? args["phase2-ep"] : 10
        ep3 = args["phase3-ep"] > 0 ? args["phase3-ep"] : 10
        ep4 = args["phase4-ep"] > 0 ? args["phase4-ep"] : 10
        ep5 = args["phase5-ep"] > 0 ? args["phase5-ep"] : 10
        ep6 = args["phase6-ep"] > 0 ? args["phase6-ep"] : 10
        return [ep1, ep2, ep3, ep4, ep5, ep6]
    end
end

function run_script_with_epochs(script_path, target_epochs, phase_name)
    println("\n" * "="^80)
    println("▶ LAUNCHING $phase_name ($target_epochs Epochs)")
    println("="^80)

    # Read original script
    content = read(script_path, String)

    # Hacky but effective epoch replacement
    # We look for standard definitions like `n_epochs = 50` or `for ep in 1:60`
    content = replace(content, r"n_epochs\s*=\s*\d+" => "n_epochs = $target_epochs")
    content = replace(content, r"epochs\s*=\s*\d+" => "epochs = $target_epochs")
    content = replace(content, r"global_epochs\s*=\s*\d+" => "global_epochs = $target_epochs")

    # Specific Phase 1 hardcoded sweeps
    content = replace(content, r"for sweep in 1:1" => "for sweep in 1:1")
    content = replace(content, r"for epoch in 1:\d+" => "for epoch in 1:$target_epochs")

    # Specific Phase 3 n_epochs = n_cycles * T_period override (we just force it physically)
    content = replace(content, r"n_epochs = n_cycles \* T_period" => "n_epochs = $target_epochs")

    # 🔥 CRITICAL SAFETY: Redirect all checkpoint saves/loads to avoid overwriting the paper's final weights.
    content = replace(content, "params_40s_" => "params_40s_auto_")

    # Save to a temporary launcher
    tmp_path = replace(script_path, ".jl" => "_auto_tmp.jl")
    write(tmp_path, content)

    try
        # Execute the temporary script
        run(`julia --project=. $tmp_path`)
        println("\n✅ $phase_name completed successfully.")
    catch e
        println("\n❌ $phase_name failed: $e")
        rethrow(e)
    finally
        # Clean up
        rm(tmp_path, force=true)
    end
end

function main()
    args = parse_commandline()
    epochs = get_target_epochs(args)

    println("🚀 INITIATING AUTOMATED CURRICULUM TRAINING PIPELINE")
    if args["smoke-test"]
        println("   MODE: SMOKE TEST (Extremely fast, logic validation only)")
    elseif args["full-run"]
        println("   MODE: FULL RUN (Deep optimization, will take massive compute time)")
    else
        println("   MODE: CUSTOM (Using command line epoch targets)")
    end

    scripts = [
        ("tests/Train/train_40_fast.jl", "Phase 1: Fast Topological Anchoring"),
        ("tests/Train/resume_phase2.jl", "Phase 2: Global Splicing"),
        ("tests/Train/resume_phase3.jl", "Phase 3: Cosine Annealing (Basin Hunting)"),
        ("tests/Train/resume_phase4.jl", "Phase 4: Deterministic Extrapolation"),
        ("tests/Train/resume_phase5.jl", "Phase 5: Deep Annealing"),
        ("tests/Train/resume_phase6.jl", "Phase 6: Ultimate Zero-Shot Floor")
    ]

    for i in 1:6
        script, name = scripts[i]
        ep = epochs[i]

        if isfile(script)
            run_script_with_epochs(script, ep, name)
        else
            println("⚠️ Warning: Script $script not found. Skipping $name.")
        end
    end

    println("\n" * "🎉"^3 * " PIPELINE EXECUTION COMPLETED! " * "🎉"^3)
    println("All resulting checkpoints are saved in `Resultados/test-8BIS/checkpoints/`.")
end

main()
