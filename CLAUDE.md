# TensorTrainStochastics.jl — notes for Claude Code

Julia package that turns José's research scripts (tensor-train representations of
stochastic spin dynamics: Glauber, Metropolis, zero-temperature, parallel update,
replica exchange / parallel tempering) into a clean, tested, documented package.
Author: José Julián Díaz Pérez (josej44), MATCOM, Universidad de La Habana.
Repo: https://github.com/josej44/TensorTrainStochastics.jl

## Environment
- Julia 1.11 (via juliaup). Always run from the repo root with `julia --project=.`
- `TensorTrains` is a LOCAL dev package: `/home/julian/.julia/dev/TensorTrains`
  (added with `Pkg.develop(path=...)`). Do not replace it with a registry version.
- The repo lives on an external disk with spaces in the path: always quote paths.
- `Manifest.toml` is not committed.
- Check after every change: `julia --project=. -e 'using TensorTrainStochastics; println("OK")'`
- Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`

## Where the original code is
The original scripts (Spanish comments, loaded with `include` from a master file):
observables.jl, plots.jl, update_rules.jl, parameters.jl, swap_criteria.jl,
monte_carlo_general.jl (Monte Carlo side) and auxiliary_tensor_train_tools.jl,
metropolis_swap_tt.jl, transition_rates_builder.jl, initialization_params.jl,
build_graph_tt.jl, graph_setup.jl, efficient_observables.jl, time_correlation.jl,
boltzmann_distribution_tt.jl, tensor_train_evolution.jl (tensor-train side).
Never edit the originals.

Location (latest version = git working tree, including uncommitted changes):
`/media/julian/Nuevo vol/UNIVERSIDAD/Investigación_Mulet/Parallel_Tempering_Italia/Codigos_version_2/MPT_Parallel_Tempering`
- `GlauberPT.jl/src/monte_carlo_efficient/` and `GlauberPT.jl/src/tensor_train_efficient/`: the scripts.
- `GlauberPT_test/*.ipynb` (and `pruebas.jl`, which is really a notebook): tests and work in progress.
- `TensorTrains.jl/`: TensorTrains v0.13.0 plus José's sum fix (commit 941abfa).
- `trash/`: superseded, ignore.
- `../Codigos/MPT_Parallel_Tempering` is an OLDER copy (still referenced by the
  global Julia 1.12 env and by `GlauberPT_test/Project.toml`); do not use it as a source.

`INVENTORY.md` maps every original function to its destination in this package.

## Architecture rules
1. `src/TensorTrainStochastics.jl` holds ALL `using`, `include` and `export`.
   Files under `src/<domain>/` only define things.
2. Organized by domain: core/, lattices/, distributions/, transitions/,
   evolution/, observables/, utils.jl (skeleton already exists, files are empty).
3. Monte Carlo code is NOT public API: it goes to `test/reference/` wrapped in
   `module ReferenceMC`, used only to validate TT results.
4. Plotting (`plots.jl`) does not go in `src/` (would make Plots a hard dependency):
   use `examples/` or a package extension later.
5. Migrate one domain per commit, in dependency order:
   core → lattices → distributions → transitions → evolution → observables.
   Compile after each file; commit with Conventional Commits (`feat:`, `fix:`,
   `refactor:`, `test:`, `docs:`, `chore:`).

## Style
- All code, comments, docstrings and messages in English.
- snake_case functions, CamelCase types, `_private` helpers, 4-space indent.
- Concrete container types (no `left = []`); no unconditional `@showprogress`
  in library functions — add a `verbose::Bool=false` keyword.
- Docstring on every exported function (signature, args, keywords, returns, example).
- Fix the typo `boltzman` → `boltzmann`.

## Conventions that must be preserved
- Spin indices 1:Q mapped to physical spins by `σ = x -> 2x - 3` (1 → -1, 2 → +1).
- Two-chain (replica) states use a combined index of size 4; the swap permutation
  is `[1, 3, 2, 4]`.
- Time units: `:sequential`/`:metropolis` update ONE site per step (operator is
  (1/N) Σᵢ Wᵢ); `:parallel` updates all sites. T_sequential ≈ N · T_parallel.
- In TensorTrains, `z` is a normalization DENOMINATOR (hence
  `multiply_by_constant!` divides `z`).

## Planned design change (do first, discuss with José)
Replace the `MCParameters` struct / NamedTuples with model types, e.g.
`ChainModel(J, h, β)` (open 1D chain, length(J) = N-1) and
`GraphModel(A, β)` (A[i,j]=J_ij, A[i,i]=h_i), both <: AbstractSpinModel.
Rate functions should take the model, not a `params` bag.

## Known bugs in the originals (fix during migration, add a test for each)
1. `distribution_b_tt`: default `swapfun = (B,s)->B` is called as `swapfun(B)`
   → MethodError. Default should be `identity`.
2. `inverse_secant_tt` calls undefined `threshold_eraser`.
3. `random_params` returns field `beta`, `parallel_random_params` returns
   `beta_1/beta_2`, but all rate functions read `params.betas` → incompatible.
4. `params.betas !== nothing ? params.betas[1] : params.betas` relies on indexing
   a Float64 with [1]; remove.
5. `full_tcorr_system` exists twice: the one in tensor_train_evolution.jl uses
   ⟨σₜ⟩² (wrong). Keep the time_correlation.jl version:
   ⟨σ₀σₜ⟩ − ⟨σ₀⟩⟨σₜ⟩.
6. `times_correlation_evolution` exists twice; the newer one leaves `B`
   undefined when `t_init ≠ 0` or when `P0` is a TensorTrain.
7. `build_sequential_transition_tt` hard-codes 2×2 identity cores although Q is
   a parameter (make Q general or assert Q == 2).
8. `glauber_transition_rate` has a TT version (4 args) and an MC version (7 args);
   keep the MC one inside `ReferenceMC`.
9. `cluster_size_prob` boundary branches may be off by one — test against
   brute-force enumeration for small N.
10. `MCParameters` asserts length(j_vector) == N-1, so it cannot describe graphs.

## Test ideas (small N, exact enumeration is cheap for N ≤ 8)
- Transition TT is stochastic: summing over the output spin gives 1 for every input.
- Boltzmann TT equals exp(-βE) by brute force.
- Evolution preserves total mass; long-time sequential Glauber/Metropolis
  converges to the Boltzmann marginals.
- Marginals sum to 1; energy from TT matches brute force.
- Inverse: ‖I − B·X‖ small.
- TT vs ReferenceMC within a few standard errors.
