# TensorTrainStochastics.jl — notes for Claude Code

Julia package for tensor-train representations of stochastic spin dynamics: Glauber,
Metropolis, zero-temperature, parallel update, replica exchange / parallel tempering.
It started as a migration of José's research scripts into a tested, documented package.
Author: José Julián Díaz Pérez (josej44), MATCOM, Universidad de La Habana.
Repo: https://github.com/josej44/TensorTrainStochastics.jl

## Status
- The migration from the original scripts is **complete** (core → lattices → distributions →
  transitions → evolution → observables, plus the Monte Carlo reference in `test/reference/`).
- The model design (`ChainModel` / `GraphModel` replacing `MCParameters` / NamedTuples) is implemented.
- `INVENTORY.md` is the record of the migration: every original function, where it went,
  renames, duplicate decisions and all bug fixes (§1.5). Keep it updated when touching migrated code.

## Environment
- Julia 1.11 locally (via juliaup); `[compat] julia = "1.10"`. CI runs 1.10, 1.12 and pre.
  Numerics can differ between versions: for numerical changes also run the suite with
  `julia +1.10` (`juliaup add 1.10`).
- Always run from the repo root with `julia --project=.`
- `TensorTrains` is a registered dependency (`[compat] TensorTrains = "0.14"`); CI
  resolves it from General. For local work the dev clone
  `/home/julian/.julia/dev/TensorTrains` (upstream main) is `Pkg.develop`-ed into the
  uncommitted `Manifest.toml` only. Never commit paths or `[sources]` in
  `Project.toml` / `test/Project.toml`. Local main differs from registered 0.14.0 only
  by `accumulate_M` normalization (harmless). TensorTrains loads MKL, so BLAS is MKL.
- The repo lives on an external disk with spaces in the path: always quote paths.
- `Manifest.toml` is not committed.
- Check after every change: `julia --project=. -e 'using TensorTrainStochastics; println("OK")'`
- Tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (about 3–5 minutes).

## Layout
- `src/TensorTrainStochastics.jl` holds ALL `using`, `include` and `export`.
  Files under `src/<domain>/` only define things.
- `src/core/`: models (`parameters.jl`), `random_chain_model` (`initialization.jl`),
  generic TT operations and inverses (`tt_utilities.jl`).
- `src/lattices/`: random graph instances. `src/distributions/`: Boltzmann TT, swap ratio,
  initial conditions. `src/transitions/`: rates, chain and graph transition operators.
  `src/evolution/`: time evolution and swap steps. `src/observables/`: marginals, correlations,
  energies, time correlations, clusters, bundles. `src/utils.jl`: empty.
- `test/runtests.jl` includes one file per domain (`test/<domain>/test_*.jl`).
- `test/helpers.jl`: brute-force references (enumeration, dense operators, exact
  distributions, `tt_value`, `tt_from_function`, `energy_difference`, …).
- `test/reference/`: `module ReferenceMC`, the Monte Carlo reference (not public API),
  used only to validate TT results.

## API conventions
- **Models.** `ChainModel(J, h, β)` (open chain, `length(J) == N-1`) and `GraphModel(A, β)`
  (`A[i,j] = J_ij`, `A[i,i] = h_i`, symmetric), both `<: AbstractSpinModel`. Accessors `nsites`,
  `local_fields`, `with_beta`; `GraphModel(::ChainModel)` gives the tridiagonal equivalent.
  Models copy their inputs and accept any β except NaN (negative β is used internally).
  Both require N ≥ 2.
- **Replicas** are two models that differ only in β (checked by `_check_replicas`), e.g.
  `boltzmann_swap_tt(m1, m2)`, `parallel_transition_tensor_train(m1, m2; …)`, `tt_swap(m1, m2, n, bond)`.
- **Rates** have the signature `rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)`
  with physical spins ±1 and the neighbourhood layout site 1 `[σ₁, σ₂]`, middle `[σᵢ₋₁, σᵢ, σᵢ₊₁]`,
  site N `[σₙ₋₁, σₙ]`. Dynamics parameters are keywords (e.g. `p0 = 0.0` for inertia), not model fields.
- **Update rules** (`build_transition_tensortrain`): `:sequential` (Glauber), `:metropolis`,
  `:zero_temp` (one random site per step, operator (1/N) Σᵢ Wᵢ) and `:parallel` (all sites, Glauber
  with inertia). T_sequential ≈ N · T_parallel. `GraphModel` supports only `:metropolis`.
- **Spin indices** 1:Q map to physical spins by `σ = x -> 2x - 3` (1 → −1, 2 → +1).
- **TT cores**: distributions `[left, right, x]`, operators `[left, right, σ, σ_next]`
  (rows sum to 1 over `σ_next`), joint two-time distributions `[left, right, σ₀, σₜ]`.
- **Two replicas** use the combined index `x + Q(y − 1)` (size 4 for Q = 2), built by
  `replica_product`; the swap x ↔ y is the permutation `[1, 3, 2, 4]`.
- **Normalization**: in TensorTrains `z` is a DENOMINATOR (`multiply_by_constant!` divides `z`).
  Observables assume a normalized distribution (`normalize!`, so z = 1 and Σ = 1).
- **Products**: pointwise product is `hadamard(A, B)` (never `Base.:*` on TT types: piracy);
  operator composition is `compose_transitions(A, B)` (A then B).
- **Inverses** (`inverse_tt`, `inverse_secant_tt`, and through `tt_swap`): iterate until the exact
  relative L2 residual is below `tol = 1e-12` (`maxsteps = 100`, `@warn` otherwise); `steps = n`
  forces a fixed count. Exact TT norms use left-orthogonalization (`_norm_l2`), not `sqrt(dot)`.
- **Errors**: invalid input throws `ArgumentError` with a clear message (wrong sizes, Q ≠ 2 where
  only Q = 2 is supported, N = 2 for functions that need N ≥ 3, unknown `update_rule`).
- **Output**: library functions never print unless `verbose = true`.

## Bugs fixed during the migration
Each has a regression test; details and evidence are in `INVENTORY.md` (CLAUDE.md list 1–10, §1.5 A–I).
- 1 `distribution_b_tt` default `swapfun`; 2 undefined `threshold_eraser`; 3, 4, 10 the params
  bags (replaced by models); 5 `full_tcorr_system` used ⟨σₜ⟩²; 6 `times_correlation_evolution`
  merged; 7 Q == 2 asserted in the sequential builder; 8 MC `glauber_transition_rate` kept in
  ReferenceMC; 9 `cluster_size_prob` boundaries verified against enumeration.
- A MC Metropolis swap sign inverted; B `EnergySwap` used a(x,y) instead of a(y,x); C
  `cluster_size_prob(B, N)`; D `sample_boltzmann_ising` missing β; E `Base.:*` piracy → `hadamard`;
  F Q = 2 assertions; G unconditional printing; H isolated graph sites; I fixed-step inverses
  (CI failure on 1.10).
- Past results affected by bug A (Metropolis swap in the MC code) are listed in `INVENTORY.md`.

## Adding new features
1. **Place** the code in the domain it belongs to (`src/<domain>/<file>.jl`); add `include`/`export`
   only in `src/TensorTrainStochastics.jl`. Keep helpers private with a leading `_`.
2. **Reuse** the existing pieces: models and `with_beta`, `replica_product`, `compose_transitions`,
   `hadamard`, `_product_tt`, `_swap_replicas`, `full_subproducts_system`. New dynamics are a rate
   function with the standard signature plus a branch in `build_transition_tensortrain`.
3. **Document** every exported function (signature, arguments, keywords, returns, example).
4. **Test** in `test/<domain>/test_*.jl`, included from `runtests.jl`:
   - exact enumeration against brute-force references in `test/helpers.jl` (cheap for N ≤ 8); always
     cover N = 2 and N = 3, and that N = 2 throws `ArgumentError` where N ≥ 3 is required;
   - stochasticity (rows sum to 1), mass conservation, Boltzmann stationarity where it applies;
   - dynamics against `ReferenceMC` within 5 standard errors when a Monte Carlo counterpart exists;
   - seeds with `StableRNG` (identical instances on every Julia version);
   - an existing test checks that no exported name clashes with TensorTrains.
5. **Never loosen a test threshold** to make it pass: find the cause first (see bug I).
   A bug found in migrated code is recorded in `INVENTORY.md` §1.5 with its test.
6. **Verify**: load check, single test file while iterating, full `Pkg.test()` before committing;
   numerical changes also on `julia +1.10`.
7. **Commit** one feature per commit with Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`,
   `docs:`, `chore:`); no Manifest, no local paths. Plotting does not go in `src/` (it would make Plots
   a hard dependency): use `examples/` or a package extension.

## Open items
- `examples/` (plots from `plots.jl`, `plot_two_Ntuples`) not migrated.
- Second-order marginals and the functions built on them reject N = 2 (could be supported).
- `random_P0` has no `rng` keyword; `build_instance` and ReferenceMC use `MersenneTwister`
  (instances differ between Julia versions).
- `inverse_tt_improve` is unexported, untested and looks suspect.
- No narrative documentation yet: the README has only the template badges and `docs/` only the
  `@autodocs` index (docstrings appear, but there is no guide or example page).

## Where the original code is (reference only)
`/media/julian/Nuevo vol/UNIVERSIDAD/Investigación_Mulet/Parallel_Tempering_Italia/Codigos_version_2/MPT_Parallel_Tempering`
(latest version = git working tree of `GlauberPT.jl/`, including uncommitted changes). Never edit
the originals. `../Codigos/MPT_Parallel_Tempering` is an older copy. The GlauberPT notebooks still
load the patched TensorTrains v0.13.0 copy in `Codigos/` through the global Julia 1.12 environment.

## Style
- All code, comments, docstrings and messages in English.
- snake_case functions, CamelCase types, `_private` helpers, 4-space indent.
- Concrete container types (no `left = []`); no unconditional `@showprogress` in library
  functions — add a `verbose::Bool=false` keyword.
