module HigherOrderDerivativesDistributionsExt

using Distributions
using HigherOrderDerivatives
using InteractiveUtils: subtypes

function _leaf_distribution_types(root)
    children = subtypes(root)
    isempty(children) && return Any[root]

    leaves = Any[]
    for child in children
        append!(leaves, _leaf_distribution_types(child))
    end

    return leaves
end

for distribution_type in _leaf_distribution_types(Distributions.ContinuousUnivariateDistribution)
    @eval function Distributions.cdf(d::$distribution_type, x::HigherOrderDerivatives.Dual)
        return HigherOrderDerivatives.Dual(
            cdf(d, x.value),
            pdf(d, x.value) * x.tangent,
        )
    end
end

end
