using Test
using Distributions
using LinearAlgebra
using HigherOrderDerivatives

@testset "HigherOrderDerivatives" begin
    @testset "scalar derivatives" begin
        cubic(x::Real) = x^3 + 2x
        @test derivative(cubic, 4.0) == 50.0
        @test derivative(x -> x^5, 2.0, 4) == 240.0

        f(x::Real) = exp(x^3 * cos(-x)^2 * sin(x))
        @test derivative(f, 0.8) ≈ 0.567190549030136 atol = 1e-12
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
