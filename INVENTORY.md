# Inventory of the original code

Source tree: `Codigos_version_2/MPT_Parallel_Tempering/` (see `CLAUDE.md` for the full path).
The latest version is the **git working tree** of `GlauberPT.jl/`, including its uncommitted changes
(zero_temp, `multi_step_transition_tt`, `distribution_b_tt_exp`, cluster functions) and the
untracked `graph_setup.jl` / `build_graph_tt.jl`.

General facts:
- `GlauberPT.jl/src.zip` is a byte-identical backup of `GlauberPT.jl/src/`.
- `GlauberPT.jl/src/GlauberPT.jl` (the module file) is broken and was not used. The `export` list is missing a
  comma after `k_step_transition_tt`, it exports the non-existent `mult_sep`, and it never includes
  `time_correlation.jl`, `graph_setup.jl` or `build_graph_tt.jl`. The notebooks `includet` the files
  one by one instead, and **none of them includes `time_correlation.jl`**.
- `GlauberPT_test/pruebas.jl` is really a notebook (JSON). Its only definition is `evolution_pair`.
- `../Codigos/MPT_Parallel_Tempering/` is an older copy of the same tree.
- Everything in `trash/` is superseded (§1.6).

## 1. Inventory

### 1.1 Tensor-train side (`GlauberPT.jl/src/tensor_train_efficient/`)

| Function | File | Purpose | Status |
|---|---|---|---|
| `identity_tensor_train(N,qs)` / `(N,qs...)` / `(A)` | auxiliary_tensor_train_tools | constant-1 TT | dup: a different implementation lives in inverse_function_test.ipynb and trash; **src is canonical** |
| `estimate_norm_tt(B)` | aux | upper bound of max\|B(x)\| | dup in trash |
| `divide_by_constant!`, `multiply_by_constant!` | aux | scale a TT through `z` (a denominator) | dup in notebooks/trash. The notebook `multiply_by_constant` (no `!`) multiplies `z`, so it actually **divides**: wrong, drop |
| `Base.:*(A::T,B::T)` | aux | Hadamard (pointwise) product | **type piracy** (TensorTrains owns both the function and the type) |
| `absorb_z_into_matrices!` | aux | spread `z` over the cores | only used in inverse_function_test.ipynb |
| `mult_sep_4`, `mult_sep_3` | aux | replica product of two TTs → joint TT on (x,y), combined index `x + Q(y-1)` | unique |
| `inverse_tt(B,bond;steps)` | aux | Newton–Schulz 1/B | dup = inverse_function_test cell 10; trash has the old `(B,steps,bond)` argument order |
| `inverse_tt_improve` | aux | Newton with per-step `normalize!`, returns `(X, z_acc)` | **unused anywhere** |
| `inverse_secant_tt` | aux | secant iteration for 1/B | **bug 2**: calls undefined `threshold_eraser`. Notebook `inverse_secant_tt_` is the same function without that line |
| `random_params`, `parallel_random_params` | initialization_params | random NamedTuple parameters | **bug 3** (`beta`/`beta_1` vs `betas`) |
| `random_P0`, `parallel_random_P0_fixed`, `transform_init_cond_simple`, `transform_init_cond_multiple` | init | initial product distributions | unique |
| `boltzman_tt(params,Q,σ;coc)` | boltzmann_distribution_tt | exp(−βE) chain TT; `coc` scales β | unique (typo) |
| `boltzman_swap_tt`, `boltzman_swap_n_tt` | boltzmann | replica ratio r(x,y) = exp((β₁−β₂)(E(x)−E(y))) and rⁿ | dup in new_reviser.ipynb (old name `boltzman_swap_tt_n`) |
| `energy_difference`, `energy_difference_n` | boltzmann | brute-force r(x,y), rⁿ | dup in new_reviser; **test reference only** |
| `tt_swap(params,n,bond)` | metropolis_swap_tt | (r+…+rⁿ)/(1+r+…+rⁿ) ≈ min(1,r) | unique |
| `build_parallel_transition_tt` | transition_rates_builder | parallel-update MPO, bond Q² | unique |
| `build_sequential_transition_tt` | trb | (1/N)ΣᵢWᵢ MPO, compressed | **bug 7** (2×2 identities hard-coded). The prototype `glauber_sequential_transition` (swap_of_energy.ipynb, undefined `bond`) is superseded |
| `build_transition_tensortrain(params; update_rule,…)` | trb | dispatcher for :sequential/:parallel/:metropolis/:zero_temp | unique |
| `parallel_transition_tensor_train` | trb | two-replica MPO (`mult_sep_4`) | builds NamedTuples with a scalar `betas`, so it relies on **bug 4** |
| `mult_sep_transition(A,B)` | trb | MPO∘MPO (A then B) | unique |
| `k_step_transition_tt`, `multi_step_transition_tt` | trb | Aᵏ by repeated product / A^(2ᵖ) by squaring | unique (`k_step` is unused) |
| `transition_rate_inertia` | trb | Glauber with inertia p0 | **bug 4** |
| `glauber_transition_rate` (4 args) | trb | Glauber | **bug 8** (the 7-arg MC version has the same name) |
| `metropolis_transition_rate`, `temp_zero_transition_rate`, `zero_dec` | trb | Metropolis / T=0 rates | unique |
| `ConstantSwap(s)`, `EnergySwap(swapTT)`, `no_swap()` | tensor_train_evolution | closures B → B after a swap | unique; `EnergySwap` is **never used** (bug B) |
| `distribution_b_tt` | tte | evolve P0 for t steps (+ swapfun) | **bug 1**: default `(B,s)->B`. The notebooks always pass `no_swap()` or `ConstantSwap` |
| `distribution_b_tt_exp` | tte | same, but squares A each step (times 1, 2, 4, …) | unique, not a duplicate |
| `full_tcorr_system(B)` | tte **and** time_correlation | ⟨σ₀σₜ⟩ and connected correlation | **dup, bug 5**: keep the time_correlation.jl version (⟨σ₀⟩⟨σₜ⟩). The notebooks ran the tte version (⟨σₜ⟩²). The two agree only when B₀ is stationary, which is how they were used |
| `times_correlation_evolution` | tte `(A, B::TensorTrain, t, bond)`; time_correlation `(A, P0::Union{…}, t, bond, t_init=0)` | joint (σ₀,σₜ) evolution | **bug 6**. These are two *methods*, not an override. The TT method is the one the notebooks used (times_correlations*.ipynb, starting from a Boltzmann B). The Union method breaks for TT input and for `t_init ≠ 0` (`t_init` is never used) |
| `full_subproducts_system`, `full_subproducts_system_2` | efficient_observables | left/right partial sums for 3-index / 4-index TTs | `_2` is the 4-index version of the same thing |
| `full_marginals_system`, `full_second_order_marginals_system` | eo | single-site / nearest-pair marginals | unique |
| `full_simple_ev_system`, `marginal_ev_system` | eo | ⟨σᵢ⟩ for one chain / two replicas | unique |
| `correlation_next_pairs_simple`, `second_moment_system`, `correlation_between_spins_system` | eo | ⟨σᵢσᵢ₊₁⟩, Pearson correlation | unique |
| `energy_function`, `energy_function_simple` | eo | ⟨E⟩ for two replicas / one chain | unique |
| `system_description`, `system_description_over_time`, `system_description_over_time_simple` | eo | bundle of observables | unique |
| `portion_of_TT`, `cluster_size_prob` | eo | fraction of sites in domains of size l | src = multiple_steps_cluster.ipynb; basic_plots.ipynb has an **older, wrong** version (no domain-wall check). **Bug 9**: the boundary branches look correct on review, but `l == N` is a BoundsError and a whole-chain domain is never counted. Test against enumeration |
| `random_3regular`, `reorder_rcm`, `add_gaussian_weights`, `build_instance` | graph_setup | random 3-regular instance → A (J off-diagonal, h on the diagonal) | unique |
| `_neighbors`, `_channels`, `_bdim`, `_encode`, `_decode`, `_metro_rate`, `_event_cores`, `_add_cores`, `build_transition_tt(A;β,bond,Q,σmap)` | build_graph_tt | sequential Metropolis MPO on a graph | unique; `s_flip = 3 - s_old` assumes Q = 2; all N events are summed before compressing |

### 1.2 Monte Carlo side (`GlauberPT.jl/src/monte_carlo_efficient/`), destined for `test/reference/ReferenceMC`

- **Parameters:** `MCParameters` (plus keyword constructor), `n_chains`, `N(params)` (a function named `N`!), `validate`, `Base.show`.
- **Driver:** `run_monte_carlo_general`, `initialize_spins`.
- **Update rules:** `glauber_transition_rate` (7 args), `compute_local_energy_change`, `parallel_update!`, `sequential_update!`,
  `sequential_update_with_energy!`, `metropolis_update!`, `metropolis_update_with_energy!`, `zero_update!`,
  `zero_update_with_energy!`, `apply_update!`.
- **Swaps:** `apply_fixed_rate_swap!`, `apply_metropolis_swap!`, `apply_metropolis_swap_with_energies!`,
  `compute_swap_acceptance_rate`, `suggest_beta_schedule`.
- **Observables:** `compute_total_energy`, `initialize_observable_accumulators`, `accumulate_observables!`, `finalize_observables`.

Notes:
- `apply_random_pair_swap!` and `estimate_acceptance_rate` sit inside string literals (dead code): drop.
- REPORTE_ERRORES.md items #1, #2, #4 and #6 are already fixed in the current tree.
- `plots.jl` (10 `plot_*` functions, identical to the trash copy) goes to `examples/`.

### 1.3 Code that only exists in notebooks

| Function | Notebook | Purpose | Destination |
|---|---|---|---|
| `evolution_pair(B,A,T)` | griffiths (+copy), pruebas.jl | lifts **any** B to the joint (x₀,x) and evolves it. This is the clean fix for bug 6 | basis of `times_correlation_evolution` |
| `mc_step_single!`, `montecarlo_magnetizations_single` | basic_plots | graph Metropolis MC (uses A) | ReferenceMC (validates the graph MPO) |
| `min_energy_config(J,H)` | basic_plots, trash/ICIMAF | exact chain ground state (dynamic programming) | `src/observables/energy.jl` as `min_energy_config(::ChainModel)` |
| `sample_boltzmann_ising(params;…)` | multiple_steps_cluster | Boltzmann samples via MC | ReferenceMC, **fix bug D** |
| `cluster_size_prob_mc(samples,l)` | multiple_steps_cluster | MC version of `cluster_size_prob` | ReferenceMC |
| `spatial_correlation(B,r)` (+ `expectation`, `potts2spin`, `twovar_expectation`) | bundell_1993 / basic_plots | C(r) via TensorTrains `twovar_marginals` | `src/observables/correlations.jl` |
| `plot_two_Ntuples` | 7 identical copies + 1 variant (bundell) | dots-vs-lines plot | `examples/` |
| `interpolar_rapido` | basic_plots | polynomial fit + plot | drop |
| `test_inverse`, `estimate_pointwise_error` | inverse_function_test | max\|B·X − 1\| on samples | `test/helpers.jl` |
| `inverse_tt_{amen,log_space,large_values,newton_optimized,geometric_series,auto}`, `inverse_logarithmic_tt`, `als_inverse_refinement`, `truncate_tensor_svd`, `evaluate_inverse` | inverse_function_test | experimental inverses | drop (they stay in the notebook) |
| `glauber_sequential_transition`, `compute_marginal_magnetization_parallel` | swap_of_energy (loads trash files) | prototypes | drop |
| `boltzman_swap_tt_n` | new_reviser | old name | drop |

### 1.4 Notebook checks that should become tests
- new_reviser: `evaluate(boltzman_swap_n_tt)` vs `energy_difference_n` on random (x,y) configurations.
- inverse_function_test: max\|B(x)·X(x) − 1\| on samples.
- multiple_steps_cluster: `cluster_size_prob(B,l)` vs `cluster_size_prob_mc`. Exact enumeration replaces sampling for N ≤ 8.
- basic_plots:
  - graph MPO magnetizations vs `montecarlo_magnetizations_single`;
  - the MPO built from `GraphModel(chain)` must equal the chain Metropolis MPO exactly.
- bundell: `full_marginals_system` / `full_second_order_marginals_system` vs TensorTrains `marginals` / `twovar_marginals`.
- times_correlations_bond_test: the Boltzmann TT is stationary, so ⟨σᵢ⟩ stays constant under evolution.

### 1.5 Bugs found during the inventory (beyond the list in CLAUDE.md; each gets a test)
- **A. The MC Metropolis swap sign is inverted** in `apply_metropolis_swap!` and `apply_metropolis_swap_with_energies!`.
  They accept with exp(−(βᵢ−βⱼ)(Eᵢ−Eⱼ)); the correct factor is exp(+(βᵢ−βⱼ)(Eᵢ−Eⱼ)).
  For a hot chain i that finds the lower energy, the correct swap is always accepted, but the code accepts it with probability < 1.
  The TT side (`boltzman_swap_tt`) has the correct sign.
- **B. `EnergySwap` (suspected).** The in-flow term `B_swap * swap` uses the acceptance a(x,y) where it needs a(y,x),
  so mass is not conserved (the following `normalize!` hides this). It has never been used.
- **C.** `cluster_size_prob(B, N)` throws a BoundsError.
- **D.** `sample_boltzmann_ising`: the field term in ΔE is missing β.
- **E.** `Base.:*` on TensorTrains is type piracy. Keep it, flag it in a comment, and consider upstreaming it.
- **F.** `full_tcorr_system`, `cluster_size_prob` and `marginal_ev_system` hard-code Q = 2. Assert it.
- **G.** `inverse_tt*` use `@show`/`@showprogress` unconditionally. Replace with `verbose::Bool=false`.

**Planned tests for A and B (exact enumeration, small N, e.g. N = 3):** one Metropolis swap step must leave
π_β₁ ⊗ π_β₂ invariant.
- ReferenceMC: build the exact 2^N × 2^N two-replica swap kernel from the acceptance rule in
  `apply_metropolis_swap_with_energies!` and check that πK = π. This should fail before the fix and pass after it.
- TT: apply `EnergySwap(tt_swap(…))` to `replica_product(boltzmann_tt(m1), boltzmann_tt(m2))`. Check mass conservation
  *before* `normalize!` and pointwise invariance. Use exact `min(1,r)` as well as the `tt_swap` approximation.
  Both tests go in with the evolution/swap domain commit (and with ReferenceMC).

**Past results to re-check (bug A).** Only these runs used `swap_criterion = :metropolis`:
- `GlauberPT.jl/src/monte_carlo_efficient/pruebas_monte_carlo.ipynb`, cell 1: `result_parallel_swap_metropolis`,
  `result_glauber_swap_metropolis`, `result_metropolis_swap_metropolis`. Parameters: N = 20, betas = [1, 2], J = 1, h = 0,
  1000 samples, T = 100.
- The same notebook also exists in `trash/monte_carlo_efficient/` and in `../Codigos/.../old/monte_carlo_efficient/` (older copies).

No `GlauberPT_test` notebook uses the Metropolis swap: every `run_monte_carlo_general` call there uses `:fixed_rate` or no swap.
No notebook calls `apply_metropolis_swap*` or `EnergySwap` directly. `tt_swap` is only checked pointwise in new_reviser
(TT sign, correct). So results from fixed-rate swap, `ConstantSwap` and TT evolution are unaffected.

### 1.6 `trash/` (drop all of it)
- Old MC: `monte_carlo_efficient/*`, `old_files/monte_carlo_*.jl`, `GlauberParams*`, `SimulationResult*`.
- Old TT builders: `tensor_builder.jl` (`tensor_b_t*`, `build_transition_tensorchain`, `transition_glauber_rate`, `mult_sep`).
- Old observables: `calculo_observables.jl`, `old_files/efficient_observables.jl`.
- `special_swap.jl` (`tensor_b_t_swap_acc_to_energy`).
- Inverse experiments: `newton_reciprocal*`, `max_tt`, `double_delta_vector`.

## 2. Mapping → destination

| Destination | Contents |
|---|---|
| `core/parameters.jl` | `AbstractSpinModel`, `ChainModel`, `GraphModel`, accessors (§5) |
| `core/initialization.jl` | `random_chain_model` (ex `random_params`) |
| `core/tt_utilities.jl` | `identity_tensor_train`, `estimate_norm_tt`, `divide_by_constant!`, `multiply_by_constant!`, `Base.:*`, `absorb_z_into_matrices!`, `replica_product` (ex `mult_sep_3/4`), `compose_transitions` (ex `mult_sep_transition`), `k_step_transition_tt`, `multi_step_transition_tt`, `inverse_tt`, `inverse_tt_improve` (unexported), `inverse_secant_tt` |
| `distributions/initial_conditions.jl` | `random_P0`, `parallel_random_P0_fixed`, `transform_init_cond_simple/_multiple`, `_product_tt(P0)` (replaces 4 inline copies) |
| `distributions/boltzmann.jl` | `boltzmann_tt`, `boltzmann_swap_tt`, `boltzmann_swap_n_tt` |
| `lattices/graph_generation.jl` | `random_3regular`, `reorder_rcm`, `build_instance` |
| `lattices/graph_weights.jl` | `add_gaussian_weights` |
| `transitions/rates.jl` | `transition_rate_inertia`, `glauber_transition_rate`, `metropolis_transition_rate`, `temp_zero_transition_rate`, `_zero_dec` |
| `transitions/chain_builder.jl` | `build_parallel_transition_tt`, `build_sequential_transition_tt`, `build_transition_tensortrain(::ChainModel)`, `parallel_transition_tensor_train` |
| `transitions/graph_builder.jl` | `_neighbors` … `_add_cores`, `build_transition_tensortrain(::GraphModel)` (ex `build_transition_tt`) |
| `evolution/evolve.jl` | `distribution_b_tt`, `distribution_b_tt_exp` |
| `evolution/swap.jl` | `ConstantSwap`, `EnergySwap`, `no_swap`, `tt_swap` |
| `observables/subproducts.jl` | `full_subproducts_system` (3-index and 4-index methods) |
| `observables/marginals.jl` | `full_marginals_system`, `full_second_order_marginals_system`, `full_simple_ev_system`, `marginal_ev_system` |
| `observables/correlations.jl` | `correlation_next_pairs_simple`, `second_moment_system`, `correlation_between_spins_system`, `spatial_correlation` |
| `observables/energy.jl` | `energy_function`, `energy_function_simple`, `min_energy_config` |
| `observables/time_correlations.jl` | `full_tcorr_system` (time_correlation.jl version), `times_correlation_evolution` (TT and P0 methods on top of the `evolution_pair` lifting; `t_init` dropped) |
| `observables/clusters.jl` | `portion_of_TT`, `cluster_size_prob` |
| `observables/system_description.jl` | `system_description`, `system_description_over_time`, `system_description_over_time_simple` |
| `test/reference/ReferenceMC.jl` | all of §1.2 (bug A fixed), adapter `MCParameters(::ChainModel; …)`, `mc_step_single!`, `montecarlo_magnetizations_single`, `sample_boltzmann_ising`, `cluster_size_prob_mc` |
| `test/helpers.jl` | `energy_difference`, `energy_difference_n`, `test_inverse`, `estimate_pointwise_error`, brute-force enumerators |
| `examples/` | `plots.jl`, `plot_two_Ntuples` |
| drop | the GlauberPT.jl module file, `src.zip`, `test.ipynb`, `parallel_random_params`, the experimental inverses, the prototypes in §1.3, all of `trash/` |

## 3. Rename table (only names that are genuinely unclear)

| Old | New | Why |
|---|---|---|
| `boltzman_*` | `boltzmann_*` | typo |
| `mult_sep_3`, `mult_sep_4` | `replica_product` (dispatch on core ndims) | the numeric suffix says nothing |
| `mult_sep_transition` | `compose_transitions` | it is MPO∘MPO, nothing is "separated" |
| `build_transition_tt(A; β)` (graph) | method `build_transition_tensortrain(::GraphModel)` | near-identical names for the same concept |
| `full_subproducts_system_2` | method of `full_subproducts_system` (4-index) | `_2` is opaque |
| `random_params` | `random_chain_model` | it now returns a model, not params |
| keyword `coc` | `beta_scale` | opaque |
| `zero_dec` | `_zero_dec` | private helper |

## 4. Duplicate decisions
- `full_tcorr_system` → time_correlation.jl version.
- `times_correlation_evolution` → merge both methods on top of the `evolution_pair` lifting.
- `cluster_size_prob` → src version (= multiple_steps_cluster).
- `inverse_secant_tt` → src version minus the `threshold_eraser` line.
- `identity_tensor_train`, `inverse_tt`, `*_by_constant!`, `boltzman_swap*`, `energy_difference*` → src versions.
- `glauber_transition_rate` → the TT version in src; the MC version goes to ReferenceMC.

## 5. Model design (approved)

```julia
abstract type AbstractSpinModel end

struct ChainModel <: AbstractSpinModel      # open 1D chain
    J::Vector{Float64}   # length N-1, J[i] couples i and i+1
    h::Vector{Float64}   # length N
    β::Float64           # any real: negative β is needed internally (boltzmann_swap_tt uses β₂-β₁)
end

struct GraphModel <: AbstractSpinModel
    A::Matrix{Float64}   # symmetric; A[i,j] = J_ij, A[i,i] = h_i
    β::Float64
end

nsites(m); fields(m); with_beta(m, β); GraphModel(m::ChainModel)
```

- **Replicas** are two models with the same couplings: `boltzmann_swap_tt(m1, m2)`, `boltzmann_swap_n_tt(m1, m2, n)`,
  `tt_swap(m1, m2, n, bond)`, `parallel_transition_tensor_train(m1, m2; …)`.
- **Rates:** `rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)`. The logic is unchanged; it reads `model.J/h/β`.
- **`p0`** is a keyword (default `0.0`) on `transition_rate_inertia` and `build_transition_tensortrain`.
- **Observables:** `energy_function(…, model)`, `energy_function_simple(bt, model)`, `system_description(B, model)`.
- **Graph builder:** `build_transition_tensortrain(::GraphModel; update_rule=:metropolis, …)`. Other rules raise an error.
- **ReferenceMC** keeps `MCParameters` unchanged, plus the adapter `MCParameters(m::ChainModel; betas=[m.β], p0, s, seed)`.
- Bugs 3, 4 and 10 disappear by construction.

## 6. TensorTrains

- `~/.julia/dev/TensorTrains` was upstream `aa04027` (v0.13.0), **without** José's fix. That fix is commit `941abfa "sum_correction"`
  in `Codigos{,_version_2}/…/TensorTrains.jl`:
  ```diff
  @@ -241,7 +241,7 @@ function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB})
       z = max(A.z, B.z)
  -    za, zb = float(A.z/z), f(float(B.z/z))
  +    za, zb = float(z/A.z), f(float(z/B.z))
  ```
- Upstream v0.14.0 (`de7acbb`, "fix sum of tensor trains #92") fixes the same bug a different way:
  `z = min(abs(A.z), abs(B.z)); za, zb = float(z/A.z), f(float(z/B.z))`. The value is the same, and since |za|, |zb| ≤ 1 it cannot overflow.
- What changed from v0.13.0 to main, for the functions we use:
  - `compress!`, `TruncBond`, `TruncBondThresh`, `normalize_eachmatrix!`, `normalize!` and the `z` field are unchanged.
  - `+`/`-` are fixed as above.
  - `*` (Hadamard) is ours, so it is unaffected.
  - `accumulate_M` now normalizes by default. This is harmless for `twovar_marginals`, which divides each pair marginal by its sum.
  - `marginals` and `twovar_marginals` now accept `F<:Number`.
  - MKL compat was widened; everything else is CI.
- Decision: depend on the registered v0.14 (`[compat] TensorTrains = "0.14"`) if it is in General, with no absolute paths
  in `test/Project.toml`. The dev clone is used only locally via `Pkg.develop`.
- The GlauberPT notebooks do **not** use the dev clone. `using TensorTrains` there resolves through the global
  `~/.julia/environments/v1.12` env, which points at `Codigos/MPT_Parallel_Tempering/TensorTrains.jl` (patched v0.13.0).
