# Predicting COVID-19 in the USA via Graph Neural ODE (WIP)

## Folders contents

- **Data**: contains files for adjacency matrix, node attributes, populations for each US state and list of time dates. Contains also the folder **Geo**, with the original data and notebook used to make the adjacency matrix;
- **Params**: pre-optimized set of parameters;
- **Train**: model_opt.jl contains the code for training the model. preprocessing.jl is a script for graph construction, dataset splitting, normalization, and covariate preparation;
- **Test**: testing.jl contains the code for uploading optimized parameters and simulate. preprocessing_testing.jl is analogous to the training preporcessing script;
- **Utils**: contains a python script to preprocess the data and the module **GraphCreator**, which contains the functions to make the GNN graph.

## Usage

Install Julia libraries with Pkg.instantiate() and run the scripts model_opt.jl or testing.jl

## Data description

Each node represents a US state and is characterized by 4 time-dependent attributes:

1. Number of new daily COVID-19 cases

2. COVID-like symptoms in the community (CLI_cmnty)

3. New daily doctor visits for flu/COVID-like symptoms

4. Attendance at indoor events with more than 10 people in the prior 24 hours

The last three variables are treated as covariates and are sourced from the COVID Trends and Impact Survey (CTIS). 
Time span: 21 May 2021 – 25 June 2022

The adjacency matrix is defined as 

\
$$A_{ij}^{dist} = P_{i}^{\alpha} P_{j}^{\beta} \exp(\frac{-d_{ij}}{r})$$


where:

- Pi​ is the population of state i
- dij is the distance between states i and j
- α, β, r are hyperparameters

Values used:

- α = β = 0.5
- r = 500 km (approximately the mean distance between US states)

Note: inside Utils/GraphCreator/data_clean.jl and in the preprocessing.jl files I select a subset of US states, right now there are 10, but to experiment faster It's better to use 3 or 4.

## Model description

The model is composed by a 2 layers Graph Neural Network with 16 hidden channels. It takes in input the number of new cases and the values of the 3 covariates. The covariates are treated as static exogenous inputs (their dynamics are not optimized), to make them continuous in time, they are interpolated using splines. It's possible to add latent variables, which initial value is initialized randomly and are optimized jointly with the network parameters. 

## Training description

The model is trained on 180 days with curriculum learning and step decaying learning rate. For each stage of the curriculum I define the epochs of training as well as the learning rate scheduling, each stage of training can stop via an early stopping criterion based on loss stagnation. The optimization algorithm is Adam. 

At the moment the data are normalized with z-score, but I'm also experimenting with Log scaling, also in this case I get underestimates in some states and overestimates in others, but at least I don't get negative values.

Work in progress...

