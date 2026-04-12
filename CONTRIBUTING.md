# Contributing

Thanks for contributing.

## Local Setup

```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Run Tests

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

## Scope

This package currently focuses on:
- scalar derivatives for scalar-input functions
- dense higher-order derivative tensors for scalar-output functions on vector inputs

If you extend the API, please update the README and add regression tests in [test/runtests.jl](/Users/geoffreykasenbacher/Projects/HigherOrderDerivatives.jl/test/runtests.jl).
