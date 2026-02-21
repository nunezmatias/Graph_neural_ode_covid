cd("Resultados/test-8BIS")
files = filter(f -> endswith(f, ".jl") && (startswith(f, "evaluate") || startswith(f, "plot") || startswith(f, "peak_shift")), readdir())
for f in files
    content = read(f, String)
    new_content = replace(content, r"params_40s_phase[2345](_ext|_final)?\.jld2" => "params_40s_phase6_final.jld2")
    if content != new_content
        write(f, new_content)
        println("Updated ", f)
    end
end
