## testing faster auto diff here
using SparseArrays
using BenchmarkTools

struct dual{N} <: Number
    f::Float64 #Union{T, NTuple{2, Dual{T}}}
    g::Array{Float64,N} #Union{T, AbstractArray}
end

struct d{N}
   A::Array{Float64,N}
end

function Base.getindex(mat::d, I...)
   #B = spzeros(size(mat.A)...)
   B = zeros(size(mat.A)...)
   B[I...] = 1.0
   return mat.A[I...], B
end

function create_d_array(x::Array{Float64})
    #m, n = size(x)
    X = d(x)
    col = Array{dual}(undef, size(x)...)
    for i in eachindex(x)
        col[i] = dual(X[i]...)
    end
    return col
end

t = rand(1000,1)

@time create_d_array(t)

import Base: +,/,*,-,^,adjoint, convert, promote_rule
import Base: +,/,*,-,^, convert, promote_rule
+(x::dual, y::dual) = dual(x.f .+ y.f, x.g .+ y.g)
+(x::dual, y::dual) = dual(x.f + y, x.g)
+(y::Real, x::dual) = dual(x.f + y, x.g)
*(x::dual, y::dual) = dual(x.f*y.f, x.f*y.g + y.f*x.g)
*(x::dual, y::Real) = dual(x.f*y, x.g*y)
*(y::Real, x::dual) = dual(x.f*y, x.g*y)

f(x) = sum(transpose(x)*x)

@time f(create_d_array(t))

@btime create_d_array(t)

f(create_d_array(t))
