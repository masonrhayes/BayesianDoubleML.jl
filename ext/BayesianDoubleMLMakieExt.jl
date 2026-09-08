# Backend-neutral Makie plotting for BayesianDoubleML.jl
#
# Loaded whenever `Makie` is in the session, whether directly or via a
# backend (`using CairoMakie` for headless/static output, `using GLMakie`
# for interactive windows). All plotting primitives used here (`Figure`,
# `Axis`, `band!`, `lines!`, `scatter!`, `text!`, `axislegend`) are generic
# Makie API, so no backend-specific code is required.

module BayesianDoubleMLMakieExt

using BayesianDoubleML
using Makie
using Polynomials
using Printf

const _SUPERSCRIPT_DIGITS = Dict(
    '0' => '⁰',
    '1' => '¹',
    '2' => '²',
    '3' => '³',
    '4' => '⁴',
    '5' => '⁵',
    '6' => '⁶',
    '7' => '⁷',
    '8' => '⁸',
    '9' => '⁹',
)

_superscript(n::Integer) = join(_SUPERSCRIPT_DIGITS[digit] for digit in string(n))

_format_magnitude(magnitude::Real, digits::Integer) =
    @sprintf("%.*f", digits, magnitude)

function _polynomial_string(p::Polynomials.Polynomial; digits::Integer = 2)
    coefficients = Polynomials.coeffs(p)
    terms = String[]
    for degree in reverse(0:(length(coefficients) - 1))
        value = round(coefficients[degree + 1]; digits = digits)
        iszero(value) && continue
        magnitude = abs(value)
        body = if degree == 0
            _format_magnitude(magnitude, digits)
        elseif magnitude == 1
            degree == 1 ? "t" : "t$(_superscript(degree))"
        else
            coefficient = _format_magnitude(magnitude, digits)
            degree == 1 ? "$(coefficient)t" : "$(coefficient)t$(_superscript(degree))"
        end
        sign = if isempty(terms)
            value < 0 ? "-" : ""
        else
            value < 0 ? " - " : " + "
        end
        push!(terms, sign * body)
    end
    return isempty(terms) ? "0" : join(terms)
end

function BayesianDoubleML.plot_exposure_response_curve(
        fitted::Union{BayesDRModel, BayesDRCurveResult};
        level = nothing,
        color = :steelblue,
        title = "Bayes-DR Exposure-Response Curve",
        xlabel = "Treatment",
        ylabel = "Expected outcome E[Y(t)]",
        figure_size = (900, 600),
        show_points::Bool = true,
        linewidth = 3,
        trend_degree::Union{Nothing, Integer} = nothing,
        trend_color = :darkorange,
        trend_linestyle = :dash,
        show_trend_equation::Bool = true,
        equation_digits::Integer = 2,
        legend_position = :rt,
    )
    if fitted isa BayesDRModel
        isfitted(fitted) || error("Model has not been fitted. Call fit!() first.")
        result = fitted.result
        result isa BayesDRCurveResult || throw(
            ArgumentError(
                "plot_exposure_response_curve is only available for " *
                    "continuous-treatment BayesDR models; use coef and confint " *
                    "for the binary-treatment ATE",
            ),
        )
    else
        result = fitted
    end

    interval_level = isnothing(level) ? result.level : level
    curve = BayesianDoubleML.exposure_response_curve(result; level = interval_level)
    order = sortperm(curve.treatment)
    treatment = curve.treatment[order]
    estimate = curve.estimate[order]
    lower = curve.lower[order]
    upper = curve.upper[order]

    figure = Figure(; size = figure_size)
    axis = Axis(
        figure[1, 1];
        title,
        xlabel,
        ylabel,
        xgridvisible = false,
        ygridcolor = (:gray, 0.15),
    )

    band!(
        axis,
        treatment,
        lower,
        upper;
        color = (color, 0.2),
        label = "$(round(Int, 100 * interval_level))% pointwise CI",
    )
    lines!(
        axis,
        treatment,
        estimate;
        color,
        linewidth,
        label = "Estimated E[Y(t)]",
    )
    if show_points
        scatter!(
            axis,
            treatment,
            estimate;
            color,
            markersize = 7,
            strokecolor = :white,
            strokewidth = 1,
        )
    end

    if trend_degree !== nothing
        degree = Int(trend_degree)
        degree >= 1 || throw(ArgumentError("trend_degree must be a positive integer"))
        length(treatment) > degree || throw(
            ArgumentError(
                "trend_degree=$degree requires more than $degree grid points " *
                    "(got $(length(treatment)))",
            ),
        )
        trend = Polynomials.fit(treatment, estimate, degree)
        dense_treatment = range(first(treatment), last(treatment); length = 400)
        lines!(
            axis,
            dense_treatment,
            trend.(dense_treatment);
            color = trend_color,
            linewidth,
            linestyle = trend_linestyle,
            label = "Degree $degree polynomial fit",
        )
        if show_trend_equation
            equation = "E[Y(t)] ≈ " * _polynomial_string(trend; digits = equation_digits)
            text!(
                axis,
                0.04,
                0.96;
                text = equation,
                space = :relative,
                align = (:left, :top),
                fontsize = 13,
            )
        end
    end

    axislegend(axis; position = legend_position, framevisible = false)

    return figure
end

end
