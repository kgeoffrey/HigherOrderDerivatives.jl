### defining diff. rules for Dual numbers ###
using Distributions, LinearAlgebra

struct Dual{T} <: Number
    f::Union{T, NTuple{2, Dual{T}}}
    g::Union{T, AbstractArray}
end

DualRealArray = Union{Dual, Real, AbstractArray}
DualandReal = Union{Dual, Real}

### Differentiation rules via overloading ###
import Base: +,/,*,-,^,adjoint, convert, promote_rule
import Base: +,/,*,-,^, convert, promote_rule
+(x::Dual, y::Dual) = Dual(x.f + y.f, x.g + y.g)
+(x::Dual, y::Real) = Dual(x.f + y, x.g)
+(y::Real, x::Dual) = Dual(x.f + y, x.g)
-(x::Dual, y::Dual) = Dual(x.f - y.f, x.g - y.g)
-(x::Dual) = Dual(-x.f, -x.g)
-(x::Dual, y::Real) = Dual(x.f -y, x.g)
-(y::Real, x::Dual) = Dual(y-x.f, -x.g)
*(x::Dual, y::Dual) = Dual(x.f*y.f, x.f*y.g + y.f*x.g)
*(x::Dual, y::Real) = Dual(x.f*y, x.g*y)
*(y::Real, x::Dual) = Dual(x.f*y, x.g*y)
/(x::Dual, y::Dual) = Dual(x.f/y.f, (y.f*x.g - x.f*y.g)/y.f^2)
/(y::Real, x::Dual) = Dual(y/x.f, (-y*x.g) / x.f^2)
/(x::Dual, y::Real) = Dual(x.f/y, x.g/y)
^(x::Dual, k::Real) = Dual(x.f^(k), (x.g * k) * x.f ^ (k-1))
^(x::Dual, k::Int) = Dual(x.f^(k), (k * x.g) * x.f ^ (k-1))
Base.exp(x::Dual) = Dual(exp(x.f), x.g * exp(x.f))
Base.sqrt(x::Dual) = Dual(sqrt(x.f), x.g / (2 * sqrt(x.f)))
Base.log(x::Dual) = Dual(log(x.f), x.g/x.f)

Base.sin(x::Dual) = Dual(sin(x.f), cos(x.f)*x.g)
Base.cos(x::Dual) = Dual(cos(x.f), -sin(x.f)*x.g)

Distributions.cdf(d, x::Dual) = Dual(cdf(d, x.f), pdf(d, x.f) * x.g)
Base.adjoint(x::Dual) = Dual(adjoint(x.f), adjoint(x.g))
LinearAlgebra.Adjoint(x::Dual) = Dual(LinearAlgebra.Adjoint(x.f), LinearAlgebra.Adjoint(x.g))
LinearAlgebra.dot(x::Dual, y::Dual) = Dual(dot(x.f,y.f), x.f * y.g + y.f * x.g)
# Base.zero(x::Dual) = Dual(zero(x.f), zero(x.g))


Base.zero(x::Dual) = zero(x.f)

Base.one(x::Dual) = one(x.f)
### conversion, promotion rules ###
convert(::Type{Dual}, x::Real) = Dual(x, one(x))
convert(::Type{Dual}, x::AbstractArray) = DualArray(x)
convert(::Type{Array}, x::Real) = [x]
Dual(x) = convert(Dual, x)
promote_rule(::Type{Dual}, ::Type{<:Number}) = Dual

function DualArray(x::AbstractArray)
    l = length(x)
    eye = I(l)
    collect = []
    for i in 1:l
        push!(collect, Dual(x[i], view(eye,i,:,)))
    end
    return collect
end


### Recursive functions for getting derivatives ###
function chain(x::DualandReal, n::Int)
    dualone = one(x)
    if n == 1
        return Dual(x, dualone)
    else
        return chain(Dual(x, dualone), n-1)
    end
end

function chain(x::AbstractArray, n::Int)
    if n == 1
        return DualArray(x)
    else
        return chain(DualArray(x), n-1)
    end
end

function dechain(x::DualandReal)
    if x isa Real
        return x
    else
        return dechain(x.g)
    end
end

function dechain(x::AbstractArray)
    if eltype(x) <: Real
        return x
    elseif x isa Dual
        return dechain(getfield(x, :g))
    elseif eltype(x) <: Vector
        return hcat(x...)
    else
        return dechain(getfield.(x, :g))
    end
end

function derivative(f::Function, x::DualRealArray, n::Int)
    return dechain(f(chain(x,n)))
end

function derivative(f::Function, x::DualRealArray)
    return dechain(f(chain(x,1)))
end

function gradient(f::Function, x::AbstractArray)
    return dechain(f(chain(x,1)))
end

function gradient(f::Function, x::AbstractArray, n::Int)
    return dechain(f(chain(x,n)))
end

function hessian(f::Function, x::AbstractArray)
    return dechain(f(chain(x, 2)))
end


### to do: change gradient function and DualArray functions (with gradients no support yet)



t = rand(4)
f(t) = sum(exp(t'*t)*2 .+ sin.(t))

hessian(f, t)

using ForwardDiff
ForwardDiff.hessian(f, t)




p = getfield.(getfield(f(chain(t, 3)), :g), :g)

Array

in(:g, fieldnames(p))

p isa Array

eltype(p) <: Real


k = getfield.(getfield(f(chain(t, 2)), :g), :g)

getfield.(hcat(k...), :g)

tt = getfield.(getfield(f(chain(t, 2)), :g), :g)
s = hcat(t...)
