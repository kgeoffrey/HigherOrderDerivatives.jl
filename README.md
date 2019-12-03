# HigherOrderDerivatives
Automatic higher order differentiation of generic julia Functions via dual numbers. 

Example:

```julia
julia> using HigherOrderDerivatives

julia> f(x::Real) = x^3 * cos(-x)^2 * sin(x);

julia> D = x -> derivative(f, x); # gives first derivative
julia> D⁹ = x -> derivative(f, x, 9); # gives 9th derivative 

julia> D(0.8)
0.567190549030136

julia> D⁹(0.8)
-2.982695656166806e6


 ```
