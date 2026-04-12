"""
    HigherOrderDerivatives

Higher-order forward-mode automatic differentiation built from recursively nested
dual numbers.

The public API currently targets:
- scalar derivatives of scalar-input functions
- dense derivative tensors of scalar-output functions on `AbstractVector` inputs
"""
module HigherOrderDerivatives

export chain, dechain, derivative, Dual, gradient, hessian

include("DualAlgebra.jl")
include("JetAlgebra.jl")

"""
    derivative(f, x)
    derivative(f, x, order)

Evaluate the first or `order`-th derivative of a scalar-input function `f` at `x`.

First derivatives use dual numbers directly. Higher-order scalar derivatives use a
truncated Taylor-series jet representation for better performance and lower
allocation pressure than repeatedly nesting scalar dual numbers.
"""
derivative(f::Function, x::Real) = scalar_derivative_via_duals(f, x)
derivative(f::Function, x::Real, order::Integer) = order == 1 ? scalar_derivative_via_duals(f, x) : scalar_derivative_via_jets(f, x, order)
derivative(f::Function, x::Dual) = scalar_derivative_via_duals(f, x)
function derivative(f::Function, x::Dual, order::Integer)
    order < 1 && throw(ArgumentError("derivative order must be at least 1"))
    if order == 1
        return scalar_derivative_via_duals(f, x)
    end

    seeded = chain(x, order)
    return dechain(active_scalar_output(f(seeded), seeded))
end

end
