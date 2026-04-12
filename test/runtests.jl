using Test
using Distributions
using LinearAlgebra
using StaticArrays
using HigherOrderDerivatives

@testset "HigherOrderDerivatives" begin
    @testset "scalar derivatives" begin
        cubic(x::Real) = x^3 + 2x
        @test derivative(cubic, 4.0) == 50.0

        @test derivative(x -> 7.0, 2.0) == 0.0
        @test derivative(x -> 7.0, 2.0, 4) == 0.0
        @test derivative(x -> x^5, 2.0, 4) == 240.0
        @test derivative(x -> x^7, 2.0, 5) == 10080.0
        @test derivative(exp, 0.3, 8) ≈ exp(0.3) atol = 1e-12
        @test derivative(sin, 0.4, 4) ≈ sin(0.4) atol = 1e-12

        f(x::Real) = exp(x^3 * cos(-x)^2 * sin(x))
        @test derivative(f, 0.8) ≈ 0.567190549030136 atol = 1e-12
    end

    @testset "jet path agrees with nested duals on smooth scalar functions" begin
        f(x::Real) = exp(x^3 * cos(x)^2 * sin(x))
        x = 0.37

        for order in 2:6
            nested = dechain(f(chain(x, order)))
            @test derivative(f, x, order) ≈ nested atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "vector derivatives" begin
        g(x) = exp(dot(x, x))
        x = ones(4)
        exp4 = exp(4.0)

        grad = gradient(g, x)
        @test grad ≈ fill(2exp4, 4)

        hess = hessian(g, x)
        expected_hess = exp4 .* (4 .* ones(4, 4) .+ 2 .* Matrix{Float64}(I, 4, 4))
        @test hess ≈ expected_hess

        third = gradient(g, x, 3)
        @test size(third) == (4, 4, 4)
        @test third[1, 1, 1] ≈ 20exp4
        @test third[1, 1, 2] ≈ 12exp4
        @test third[1, 2, 3] ≈ 8exp4

        @test gradient(_ -> 5.0, x) == zeros(4)
        @test hessian(_ -> 5.0, x) == zeros(4, 4)
    end

    @testset "small-vector tangents use StaticArrays" begin
        small = chain([1.0, 2.0, 3.0], 1)
        large = chain(collect(1.0:9.0), 1)

        @test small[1].tangent isa SVector{3,Float64}
        @test large[1].tangent isa Vector{Float64}
    end

    @testset "polynomial tensor structure" begin
        h(x) = x[1]^3 + x[1] * x[2]^2
        x = [2.0, 3.0]

        @test gradient(h, x) == [21.0, 12.0]
        @test hessian(h, x) == [12.0 6.0; 6.0 4.0]

        third = gradient(h, x, 3)
        expected = zeros(2, 2, 2)
        expected[1, 1, 1] = 6.0
        expected[1, 2, 2] = 2.0
        expected[2, 1, 2] = 2.0
        expected[2, 2, 1] = 2.0
        @test third == expected
    end

    @testset "clear errors for unsupported vector-output functions" begin
        bad(x) = [x[1]^2, x[2]^2]
        @test_throws ArgumentError gradient(bad, [1.0, 2.0])
    end

    @testset "optional Distributions extension" begin
        normal = Normal()
        @test derivative(x -> cdf(normal, x), 0.0) ≈ pdf(normal, 0.0)
    end
end
