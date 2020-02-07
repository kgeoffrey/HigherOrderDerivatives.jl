## testing faster auto diff here
using SparseArrays
using BenchmarkTools

struct dual{T, N} <: Number
    f::Union{T, NTuple{2, Dual{T}}}
    g::Union{T, Array{T,N}}
end

struct d{N}
   A::Array{Float64,N}
end

function Base.getindex(mat::d, I...)
   #B = spzeros(size(mat.A)...)
   B = zeros(size(mat.A)...)
   B[I...] = 1.0
   return dual([mat.A[I...]], B)
end

function create_d_array(x::Array{Float64})
    X = d(x)
    col = Array{dual}(undef, size(x)...)
    for i in eachindex(x)
        col[i] = X[i]
    end
    return col
end



t = rand(1000, 3)
@time pre = create_d_array(t)

import Base: +,/,*,-,^,adjoint, convert, promote_rule
import Base: +,/,*,-,^, convert, promote_rule
+(x::dual, y::dual) = dual(x.f .+ y.f, x.g .+ y.g)
+(x::dual, y::Real) = dual(x.f .+ y, x.g)
+(y::Real, x::dual) = dual(x.f .+ y, x.g)
*(x::dual, y::dual) = dual(x.f .* y.f, x.f .* y.g .+ y.f .* x.g)

*(x::dual, y::Real) = dual(x.f .* y, x.g .* y)
*(y::Real, x::dual) = dual(x.f .* y, x.g .* y)



f(x) = transpose(x) * x

using BenchmarkTools
using ForwardDiff
import HigherOrderDerivatives

t = rand(1000, 3)

@btime y = create_d_array(t)
# @btime f(y).g



@btime ForwardDiff.gradient(f, t)

### Test something new ######################################
