# HigherOrderDerivatives.jl

Higher-order forward-mode automatic differentiation for Julia functions using recursively nested dual numbers.

The package is intentionally small and predictable:
- scalar derivatives for scalar-input functions via `derivative`
- gradients, Hessians, and dense higher-order derivative tensors for scalar-output functions on vector inputs via `gradient` and `hessian`

Under the hood:
- first-order scalar derivatives use dual numbers
- higher-order scalar derivatives use truncated Taylor jets
- small vector tangents use `StaticArrays` to reduce allocation pressure

## Installation

```julia
pkg> add HigherOrderDerivatives
```

The core package depends only on Julia's standard `LinearAlgebra` library.

If you also want automatic differentiation through `Distributions.cdf`, install `Distributions` as well:

```julia
pkg> add Distributions
```

The extension will load automatically when both packages are present.

## Quick Start

```julia
julia> using HigherOrderDerivatives

julia> f(x::Real) = exp(x^3 * cos(-x)^2 * sin(x))

julia> derivative(f, 0.8)
0.567190549030136

julia> derivative(f, 0.8, 9)
-2.982695656166806e6
```

```julia
julia> using HigherOrderDerivatives, LinearAlgebra

julia> g(x::AbstractVector) = exp(dot(x, x))

julia> x = ones(4)
4-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0

julia> gradient(g, x)
4-element Vector{Float64}:
 109.19630006628847
 109.19630006628847
 109.19630006628847
 109.19630006628847

julia> hessian(g, x)
4×4 Matrix{Float64}:
 327.589  218.393  218.393  218.393
 218.393  327.589  218.393  218.393
 218.393  218.393  327.589  218.393
 218.393  218.393  218.393  327.589

julia> size(gradient(g, x, 3))
(4, 4, 4)
```

## Optional `Distributions` Integration

```julia
julia> using HigherOrderDerivatives, Distributions

julia> derivative(x -> cdf(Normal(), x), 0.0)
0.3989422804014327
```

At the moment, the optional `Distributions.cdf` extension is only implemented for first-order differentiation.

## API

```julia
derivative(f, x)
derivative(f, x, order)
gradient(f, x)
gradient(f, x, order)
hessian(f, x)
```

`gradient(f, x, order)` returns the dense `order`-th derivative tensor.

## Scope and Limitations

- Vector input support currently targets `AbstractVector` only.
- Vector APIs assume `f(x)` is scalar-valued.
- Higher-order tensors scale quickly in both memory and runtime as input dimension and derivative order increase.
- The implementation is designed to be small, readable, and dependable rather than competitive with industrial AD systems on large problems.
- Very high-order scalar derivatives of arbitrary functions are supported through the jet backend, but optional library extensions may still provide only first-order rules.

## Development

Run the test suite with:

```julia
julia --project -e 'using Pkg; Pkg.test()'
```
