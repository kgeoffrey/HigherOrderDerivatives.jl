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
+(x::Dual, y::Dual) = Dual(x.f .+ y.f, x.g .+ y.g)
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

Base.abs(x::Dual) = sqrt(abs2(x))
Base.abs2(x::Dual) = real(conj(x)*x)
Base.conj(x::Dual) = Dual(conj(x.f) , conj(x.g))
Base.real(x::Dual) = Dual(real(x.f) , real(x.g))

Distributions.cdf(d, x::Dual) = Dual(cdf(d, x.f), pdf(d, x.f) * x.g)
Base.adjoint(x::Dual) = Dual(adjoint(x.f), adjoint(x.g))
LinearAlgebra.Adjoint(x::Dual) = Dual(LinearAlgebra.Adjoint(x.f), LinearAlgebra.Adjoint(x.g))
LinearAlgebra.dot(x::Dual, y::Dual) = Dual(dot(x.f,y.f), x.f * y.g + y.f * x.g)
# Base.zero(x::Dual) = Dual(zero(x.f), zero(x.g))

Base.isless(z::Dual,w::Dual) = z.f < w.f
Base.isless(z::Real,w::Dual) = z < w.f
Base.isless(z::Dual,w::Real) = z.f < w

Base.zero(x::Dual) = zero(x.f)

Base.one(x::Dual) = one(x.f)
### conversion, promotion rules ###
convert(::Type{Dual}, x::Real) = Dual(x, one(x))
convert(::Type{Dual}, x::AbstractArray) = DualArray(x)
convert(::Type{Array}, x::Real) = [x]
Dual(x) = convert(Dual, x)
Dual(x::Dual) = Dual(x, one(x.f))
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

function dechain(x::DualRealArray, dim::Int)
    if x isa Dual
        return dechain(getfield(x, :g), dim)
    elseif eltype(x) <: Real
        l = size(x, 1)
        ar = Int.(ones(dim).*l)
        x = reshape(x, tuple(ar...)...)
        return x
    elseif eltype(x) <: Dual
        return dechain(getfield.(x, :g), dim)
    elseif eltype(x) <: Vector
        x = cat(x...;dims = dim)
        return dechain(x, dim)
    else
        return dechain(getfield.(x, :g), dim)
    end
end

derivative(f::Function, x::DualandReal, n::Int) = dechain(f.(chain([x],n)), 1)[1]
derivative(f::Function, x::DualandReal) = dechain(f.(chain([x],1)), 1)[1]
gradient(f::Function, x::AbstractArray) = dechain(f(chain(x,1)), 1)
gradient(f::Function, x::AbstractArray, n::Int) = dechain(f(chain(x, n)), n)
hessian(f::Function, x::AbstractArray) = dechain(f(chain(x, 2)), 2)

## to do, add support for multi dimensional arrays (change dual array function)

using ForwardDiff
using BenchmarkTools

function fillerq(x::AbstractArray)

    m, n  = size(x)
    collect = Array{Dual,2}(undef, m, n)

    A = zeros(Float64, m, n)

    for i in eachindex(x)
        addone!(A, i)
        collectt!(collect, x, i, A)
        setzero!(A, i)
    end

    return collect
    #return collect
end


function addone!(A, i)
    A[i] = 1.0
    return A
end

function collectt!(mat, x, i, A)
    mat[i] = Dual(x[i], A)
    return collect
end

function setzero!(A, i)
    A[i] = 0.0
    return A
end


t = rand(2, 2)

@btime fillerq(t)


pe = fillerq(t)

sum(pe[1].g)


f(x) = sum((x).*x)


##################

using SparseArrays

function pepe(x::AbstractArray)
    m, n  = size(x)
    col = Array{Dual}(undef, m, n)
    #col = zeros(Dual, m, n)
    A = zeros(m, n)
    #D(f,g) = Dual(f, g)
    for i in eachindex(col)
        #A = zeros(m, n)
        A[i] = 1.0
        #print(A)
        col[i] = Dual(x[i], copy(A))
        A[i] = 0
    end

    return col
end

t = rand(1000)
@time s = pepe(t)

struct Store{N}
   A::Array{Float64,N}
end

function Base.getindex(store::Store, I...)
   #B = spzeros(size(store.A)...)
   B = zeros(size(store.A)...)
   B[I...] = 1.0
   return store.A[I...], B
end

@time n = Store(t)



function hm(x)
    #m, n = size(x)
    X = Store(x)
    col = Array{Dual}(undef, size(x)...)
    for i in eachindex(x)
        col[i] = Dual(X[i]...)
    end
    return col
end

@time s = hm(t)




f(x) = sum(transpose(x)*x)

@time f(s).g

using ForwardDiff

@time ForwardDiff.gradient(f, t)


@time 12000. * rand(100)

@time [12000.0] .* rand(100)
