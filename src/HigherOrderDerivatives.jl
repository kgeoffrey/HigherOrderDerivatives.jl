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

end
