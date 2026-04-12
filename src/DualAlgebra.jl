using LinearAlgebra

struct Dual{P,T} <: Real
    value::P
    tangent::T
end

const ScalarInput = Union{Dual,Real}

primal(x::Real) = x
primal(x::Dual) = primal(x.value)

zerolike(x::Real) = zero(x)
zerolike(x::AbstractArray) = map(zerolike, x)
zerolike(x::Dual) = Dual(zerolike(x.value), zerolike(x.tangent))

onelike(x::Real) = one(x)
onelike(x::Dual) = Dual(onelike(x.value), zerolike(x.tangent))

Base.zero(x::Dual) = zerolike(x)
Base.one(x::Dual) = onelike(x)
Base.abs(x::Dual) = sqrt(abs2(x))
Base.abs2(x::Dual) = real(conj(x) * x)
Base.conj(x::Dual) = Dual(conj(x.value), conj(x.tangent))
Base.real(x::Dual) = Dual(real(x.value), real(x.tangent))
Base.adjoint(x::Dual) = Dual(adjoint(x.value), adjoint(x.tangent))
Base.show(io::IO, x::Dual) = print(io, "Dual(value=", x.value, ", tangent=", x.tangent, ")")

Base.isless(x::Dual, y::Dual) = x.value < y.value
Base.isless(x::Real, y::Dual) = x < y.value
Base.isless(x::Dual, y::Real) = x.value < y

import Base: +, -, *, /, ^

+(x::Dual, y::Dual) = Dual(x.value + y.value, x.tangent + y.tangent)
+(x::Dual, y::Real) = Dual(x.value + y, x.tangent)
+(x::Real, y::Dual) = y + x

-(x::Dual, y::Dual) = Dual(x.value - y.value, x.tangent - y.tangent)
-(x::Dual, y::Real) = Dual(x.value - y, x.tangent)
-(x::Real, y::Dual) = Dual(x - y.value, -y.tangent)
-(x::Dual) = Dual(-x.value, -x.tangent)

*(x::Dual, y::Dual) = Dual(x.value * y.value, x.value * y.tangent + y.value * x.tangent)
*(x::Dual, y::Real) = Dual(x.value * y, x.tangent * y)
*(x::Real, y::Dual) = y * x

/(x::Dual, y::Dual) = Dual(x.value / y.value, (y.value * x.tangent - x.value * y.tangent) / y.value^2)
/(x::Dual, y::Real) = Dual(x.value / y, x.tangent / y)
/(x::Real, y::Dual) = Dual(x / y.value, (-x * y.tangent) / y.value^2)

^(x::Dual, k::Integer) = Dual(x.value^k, (k * x.tangent) * x.value^(k - 1))
^(x::Dual, k::Real) = Dual(x.value^k, (k * x.tangent) * x.value^(k - 1))

Base.exp(x::Dual) = begin
    fx = exp(x.value)
    Dual(fx, x.tangent * fx)
end

Base.sqrt(x::Dual) = begin
    root = sqrt(x.value)
    Dual(root, x.tangent / (2 * root))
end

Base.log(x::Dual) = Dual(log(x.value), x.tangent / x.value)
Base.sin(x::Dual) = Dual(sin(x.value), cos(x.value) * x.tangent)
Base.cos(x::Dual) = Dual(cos(x.value), -sin(x.value) * x.tangent)

function LinearAlgebra.dot(x::AbstractVector{<:Dual}, y::AbstractVector{<:Dual})
    axes(x) == axes(y) || throw(DimensionMismatch("vectors must have matching axes"))
    isempty(x) && throw(ArgumentError("dot requires a non-empty vector"))

    accumulator = x[firstindex(x)] * y[firstindex(y)]
    for index in Iterators.drop(eachindex(x, y), 1)
        accumulator += x[index] * y[index]
    end

    return accumulator
end

"""
    variable(x)

Seed `x` as an active differentiation variable with tangent one.
"""
function variable(x::ScalarInput)
    return Dual(x, one(x))
end

function unit_tangent(sample, length::Int, active::Int)
    basis = Vector{typeof(one(sample))}(undef, length)
    off_diagonal = zero(sample)
    on_diagonal = one(sample)

    for index in 1:length
        basis[index] = index == active ? on_diagonal : off_diagonal
    end

    return basis
end

function DualArray(x::AbstractVector)
    return [Dual(x[index], unit_tangent(x[index], length(x), index)) for index in eachindex(x)]
end

"""
    chain(x, order)

Wrap `x` in `order` layers of dual numbers so evaluating a function on the result
tracks derivatives up to that order.
"""
function chain(x::ScalarInput, order::Integer)
    order < 1 && throw(ArgumentError("derivative order must be at least 1"))

    seeded = variable(x)
    for _ in 2:order
        seeded = variable(seeded)
    end

    return seeded
end

function chain(x::AbstractVector, order::Integer)
    order < 1 && throw(ArgumentError("derivative order must be at least 1"))

    seeded = DualArray(x)
    for _ in 2:order
        seeded = DualArray(seeded)
    end

    return seeded
end

dechain(x::Real) = x
dechain(x::Dual) = dechain(x.tangent)

function require_scalar_output(result, api_name::AbstractString)
    if result isa ScalarInput
        return result
    end

    throw(ArgumentError("`$api_name` expects a scalar-output function when differentiating vector inputs"))
end

function dechain(x::Union{Dual,AbstractArray}, dim::Integer)
    dim < 1 && throw(ArgumentError("tensor order must be at least 1"))
    return tensorize(x, dim)
end

tensorize(x::Dual, dim::Integer) = tensorize(x.tangent, dim)

function tensorize(x::AbstractVector, dim::Integer)
    if dim == 1
        return collect(x)
    end

    slices = map(entry -> tensorize(entry, dim - 1), x)
    return cat(slices...; dims=dim)
end

"""
    derivative(f, x)
    derivative(f, x, order)

Evaluate the first or `order`-th derivative of a scalar-input function `f` at `x`.
"""
derivative(f::Function, x::ScalarInput) = dechain(f(chain(x, 1)))
derivative(f::Function, x::ScalarInput, order::Integer) = dechain(f(chain(x, order)))

"""
    gradient(f, x)
    gradient(f, x, order)

Evaluate the gradient or dense `order`-th derivative tensor of a scalar-output
function `f` on vector input `x`.
"""
gradient(f::Function, x::AbstractVector) = dechain(require_scalar_output(f(chain(x, 1)), "gradient"), 1)
gradient(f::Function, x::AbstractVector, order::Integer) = dechain(require_scalar_output(f(chain(x, order)), "gradient"), order)

"""
    hessian(f, x)

Evaluate the Hessian matrix of a scalar-output function `f` on vector input `x`.
"""
hessian(f::Function, x::AbstractVector) = dechain(require_scalar_output(f(chain(x, 2)), "hessian"), 2)
