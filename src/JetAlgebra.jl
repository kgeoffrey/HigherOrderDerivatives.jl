using StaticArrays

struct Jet{T,M} <: Real
    coeffs::SVector{M,T}
end

jet_order(::Jet{T,M}) where {T,M} = M - 1
jet_value(x::Jet) = x.coeffs[1]

function constant_jet(x::T, ::Val{N}) where {T,N}
    return Jet(SVector{N + 1,T}(ntuple(index -> index == 1 ? x : zero(x), N + 1)))
end

function variable_jet(x::T, ::Val{N}) where {T,N}
    return Jet(SVector{N + 1,T}(ntuple(index -> index == 1 ? x : index == 2 ? one(x) : zero(x), N + 1)))
end

Base.zero(x::Jet) = constant_jet(zero(jet_value(x)), Val(jet_order(x)))
Base.one(x::Jet) = constant_jet(one(jet_value(x)), Val(jet_order(x)))
Base.isless(x::Jet, y::Jet) = jet_value(x) < jet_value(y)
Base.isless(x::Real, y::Jet) = x < jet_value(y)
Base.isless(x::Jet, y::Real) = jet_value(x) < y
Base.show(io::IO, x::Jet) = print(io, "Jet(coeffs=", x.coeffs, ")")

function constant_like(x::Jet, y::S) where {S<:Real}
    return constant_jet(y, Val(jet_order(x)))
end

function jet_output(result::Jet{T,M}, ::Val{N}) where {T,M,N}
    M == N + 1 || throw(ArgumentError("jet order mismatch"))
    return result
end

function jet_output(result::Real, order::Val{N}) where {N}
    return constant_jet(result, order)
end

import Base: +, -, *, /, ^, exp, log, sin, cos, sqrt

+(x::Jet{T,M}, y::Jet{S,M}) where {T,S,M} = Jet(x.coeffs + y.coeffs)
+(x::Jet, y::Real) = x + constant_like(x, y)
+(x::Real, y::Jet) = y + x

-(x::Jet{T,M}, y::Jet{S,M}) where {T,S,M} = Jet(x.coeffs - y.coeffs)
-(x::Jet, y::Real) = x - constant_like(x, y)
-(x::Real, y::Jet) = constant_like(y, x) - y
-(x::Jet) = Jet(-x.coeffs)

function *(x::Jet{T,M}, y::Jet{S,M}) where {T,S,M}
    P = promote_type(T, S)
    order = M - 1
    coeffs = MVector{M,P}(undef)

    for degree in 0:order
        acc = zero(P)
        for left_degree in 0:degree
            right_degree = degree - left_degree
            acc += x.coeffs[left_degree + 1] * y.coeffs[right_degree + 1]
        end
        coeffs[degree + 1] = acc
    end

    return Jet(SVector(coeffs))
end

*(x::Jet, y::Real) = x * constant_like(x, y)
*(x::Real, y::Jet) = y * x

function /(x::Jet{T,M}, y::Jet{S,M}) where {T,S,M}
    P = promote_type(T, S)
    order = M - 1
    coeffs = MVector{M,P}(undef)
    coeffs[1] = x.coeffs[1] / y.coeffs[1]

    for degree in 1:order
        acc = x.coeffs[degree + 1]
        for offset in 1:degree
            acc -= y.coeffs[offset + 1] * coeffs[degree - offset + 1]
        end
        coeffs[degree + 1] = acc / y.coeffs[1]
    end

    return Jet(SVector(coeffs))
end

/(x::Jet, y::Real) = x / constant_like(x, y)
/(x::Real, y::Jet) = constant_like(y, x) / y

^(x::Jet, power::Integer) = power < 0 ? one(x) / (x ^ (-power)) : _pow_integer(x, power)
^(x::Jet, power::Real) = exp(power * log(x))

function _pow_integer(x::Jet, power::Integer)
    result = one(x)
    base = x
    remaining = power

    while remaining > 0
        if isodd(remaining)
            result *= base
        end
        remaining ÷= 2
        remaining == 0 && break
        base *= base
    end

    return result
end

function exp(x::Jet{T,M}) where {T,M}
    order = M - 1
    coeffs = MVector{M,T}(undef)
    coeffs[1] = exp(x.coeffs[1])

    for degree in 1:order
        acc = zero(T)
        for offset in 1:degree
            acc += offset * x.coeffs[offset + 1] * coeffs[degree - offset + 1]
        end
        coeffs[degree + 1] = acc / degree
    end

    return Jet(SVector(coeffs))
end

function log(x::Jet{T,M}) where {T,M}
    order = M - 1
    coeffs = MVector{M,T}(undef)
    coeffs[1] = log(x.coeffs[1])

    for degree in 1:order
        acc = degree * x.coeffs[degree + 1]
        for offset in 1:(degree - 1)
            acc -= (degree - offset) * x.coeffs[offset + 1] * coeffs[degree - offset + 1]
        end
        coeffs[degree + 1] = acc / (degree * x.coeffs[1])
    end

    return Jet(SVector(coeffs))
end

function sqrt(x::Jet{T,M}) where {T,M}
    order = M - 1
    coeffs = MVector{M,T}(undef)
    coeffs[1] = sqrt(x.coeffs[1])

    for degree in 1:order
        acc = x.coeffs[degree + 1]
        for offset in 1:(degree - 1)
            acc -= coeffs[offset + 1] * coeffs[degree - offset + 1]
        end
        coeffs[degree + 1] = acc / (2 * coeffs[1])
    end

    return Jet(SVector(coeffs))
end

function sincos(x::Jet{T,M}) where {T,M}
    order = M - 1
    sin_coeffs = MVector{M,T}(undef)
    cos_coeffs = MVector{M,T}(undef)
    sin_coeffs[1] = sin(x.coeffs[1])
    cos_coeffs[1] = cos(x.coeffs[1])

    for degree in 1:order
        sin_acc = zero(T)
        cos_acc = zero(T)

        for offset in 1:degree
            sin_acc += offset * x.coeffs[offset + 1] * cos_coeffs[degree - offset + 1]
            cos_acc -= offset * x.coeffs[offset + 1] * sin_coeffs[degree - offset + 1]
        end

        sin_coeffs[degree + 1] = sin_acc / degree
        cos_coeffs[degree + 1] = cos_acc / degree
    end

    return Jet(SVector(sin_coeffs)), Jet(SVector(cos_coeffs))
end

sin(x::Jet) = first(sincos(x))
cos(x::Jet) = last(sincos(x))

function scalar_derivative_via_jets(f::Function, x::Real, order::Integer)
    order < 1 && throw(ArgumentError("derivative order must be at least 1"))

    result = if order <= 16
        _scalar_derivative_via_jets(f, x, Val(order))
    else
        _scalar_derivative_via_jets_dynamic(f, x, order)
    end

    return result
end

function _scalar_derivative_via_jets(f::Function, x::Real, order::Val{N}) where {N}
    seeded = variable_jet(x, order)
    result = jet_output(f(seeded), order)
    scale = oftype(jet_value(result), factorial(big(N)))
    return scale * result.coeffs[N + 1]
end

function _scalar_derivative_via_jets_dynamic(f::Function, x::Real, order::Integer)
    seeded = chain(x, order)
    result = active_scalar_output(f(seeded), seeded)
    return dechain(result)
end
