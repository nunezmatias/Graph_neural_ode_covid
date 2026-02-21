using NPZ

data = npzread("Data/data_filtered.npz")
state = "NY"
state_data = data[state]

println("Data for $state:")
println("Channel 1 (Cases): ", state_data[1, [1, 50, 100, 200, 300]])
println("Channel 2 (CLI?): ", state_data[2, [1, 50, 100, 200, 300]])
println("Channel 3 (Visits?): ", state_data[3, [1, 50, 100, 200, 300]])
println("Channel 4 (Indoor?): ", state_data[4, [1, 50, 100, 200, 300]])

# Check if trend is increasing or decreasing with Cases
# If Channel 2, 3, 4 are high when cases are high -> Alarm Signals
