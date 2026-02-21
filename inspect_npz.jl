using NPZ

data = npzread("Data/data_filtered.npz")
println("Keys: ", keys(data))

# Check one state
first_key = first(keys(data))
first_val = data[first_key]
println("Shape of $first_key: ", size(first_val))

# Start checking if we can infer content from range/magnitude
println("\nFirst 5 days for $first_key:")
println(first_val[:, 1:5])
