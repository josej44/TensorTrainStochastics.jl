# Bundles of observables for two-replica (and single-chain) distributions.

"""
    system_description(B, model::ChainModel)

All equal-time observables of a normalized two-replica distribution `B`, as a NamedTuple:
`full_marginals`, `full_second_order_marginals`, `full_expected_values`,
`full_second_moments`, `full_correlations`, `energy` (one entry per replica where
applicable). Requires `N ≥ 3`.

# Example
```julia
obs = system_description(B12, m1)
obs.energy
```
"""
function system_description(B, model::ChainModel)
    full_prod_sys = full_subproducts_system(B)
    full_marginals = full_marginals_system(B, full_prod_sys, true)
    full_second_order_marginals = full_second_order_marginals_system(B, full_prod_sys, true)
    full_expected_values = marginal_ev_system(full_marginals)
    full_second_moments = second_moment_system(full_second_order_marginals)
    full_correlations = correlation_between_spins_system(full_second_moments, full_expected_values)
    energy = energy_function(full_expected_values, full_second_moments, model)

    observables_system = (
        full_marginals = full_marginals,
        full_second_order_marginals = full_second_order_marginals,
        full_expected_values = full_expected_values,
        full_second_moments = full_second_moments,
        full_correlations = full_correlations,
        energy = energy
    )

    return observables_system
end

"""
    system_description_over_time(B_t, model::ChainModel)

[`system_description`](@ref) for every distribution in `B_t`, as a NamedTuple of vectors:
`full_marginals_t`, `full_second_order_marginals_t`, `full_expected_values_t`,
`full_second_moments_t`, `full_correlations_t`, `energy_t`.

# Example
```julia
obs_t = system_description_over_time(distribution_b_tt(A12, P0, 100, 10), m1)
```
"""
function system_description_over_time(B_t, model::ChainModel)
    observables = [system_description(B, model) for B in B_t]
    observables_over_time = (
        full_marginals_t = [o.full_marginals for o in observables],
        full_second_order_marginals_t = [o.full_second_order_marginals for o in observables],
        full_expected_values_t = [o.full_expected_values for o in observables],
        full_second_moments_t = [o.full_second_moments for o in observables],
        full_correlations_t = [o.full_correlations for o in observables],
        energy_t = [o.energy for o in observables]
    )
    return observables_over_time
end

"""
    system_description_over_time_simple(B_t, model::ChainModel)

Single-chain version: NamedTuple with `full_expected_values_t` (magnetizations per site)
and `energy_t` for every distribution in `B_t`. Requires `N ≥ 3`.

# Example
```julia
obs_t = system_description_over_time_simple(distribution_b_tt(A, P0, 100, 10), m)
```
"""
function system_description_over_time_simple(B_t, model::ChainModel)
    observables_over_time = (
        full_expected_values_t = [full_simple_ev_system(B) for B in B_t],
        energy_t = [energy_function_simple(B, model) for B in B_t]
    )
    return observables_over_time
end
