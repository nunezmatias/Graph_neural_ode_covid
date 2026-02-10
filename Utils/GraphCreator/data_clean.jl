module_dir = @__DIR__

# states = ["WV","FL","IL","MN","MD","RI","ID","NH","NC","VT","CT","DE","NM","CA","NJ","WI","OR","NE","PA","WA","LA",
#           "GA","AL","UT","OH","TX","CO","SC","OK","TN","WY","ND","KY","ME","NY","NV","MI","AR","MS","MO",
#           "MT","KS","IN","SD","MA","VA","DC","IA","AZ"]

# filtered states smaller subset
states = ["FL","IL","NC","CA","NJ","GA","OH","TX","NY","VA"]

features_original = NPZ.npzread(joinpath(module_dir, "../../Data/data_filtered.npz"))

# select states
features = Dict(k => features_original[k] for k in states)










