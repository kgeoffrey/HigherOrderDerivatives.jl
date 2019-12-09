# HigherOrderDerivatives
Automatic higher order differentiation of generic julia Functions via dual numbers.

Install the package by using the Julia package manager:
```julia
Pkg.add("HigherOrderDerivatives")
```
## Example:

```julia
julia> using HigherOrderDerivatives

julia> f(x::Real) = exp(x^3 * cos(-x)^2 * sin(x));

julia> D = x -> derivative(f, x); # gives first derivative
julia> D⁹ = x -> derivative(f, x, 9); # gives 9th derivative

julia> D(0.8)
0.567190549030136

julia> D⁹(0.8)
-2.982695656166806e6

julia> g(x::Vector) = exp(x'*x)

julia> test = ones(4)

julia> grad = x -> gradient(g, x); # gives the gradient
julia> hess = x -> hessian(g, x); # hessian
julia> higher_grad = x,y -> gradient(g, x, y) ## gives the yth order gradient

julia> higher_grad(test, 3)

4x4x4 Array{Float64,3}:
[1091.9630006628847 655.1778003977308 655.1778003977308 655.1778003977308;
655.1778003977308 655.1778003977308 436.7852002651539 436.7852002651539;
655.1778003977308 436.7852002651539 655.1778003977308 436.7852002651539;
655.1778003977308 436.7852002651539 436.7852002651539 655.1778003977308]

[655.1778003977308 655.1778003977308 436.7852002651539 436.7852002651539;
655.1778003977308 1091.9630006628847 655.1778003977308 655.1778003977308;
436.7852002651539 655.1778003977308 655.1778003977308 436.7852002651539;
436.7852002651539 655.1778003977308 436.7852002651539 655.1778003977308]

[655.1778003977308 436.7852002651539 655.1778003977308 436.7852002651539;
436.7852002651539 655.1778003977308 655.1778003977308 436.7852002651539;
655.1778003977308 655.1778003977308 1091.9630006628847 655.1778003977308;
436.7852002651539 436.7852002651539 655.1778003977308 655.1778003977308]

[655.1778003977308 436.7852002651539 436.7852002651539 655.1778003977308;
436.7852002651539 655.1778003977308 436.7852002651539 655.1778003977308;
436.7852002651539 436.7852002651539 655.1778003977308 655.1778003977308;
655.1778003977308 655.1778003977308 655.1778003977308 1091.9630006628847]

 ```
