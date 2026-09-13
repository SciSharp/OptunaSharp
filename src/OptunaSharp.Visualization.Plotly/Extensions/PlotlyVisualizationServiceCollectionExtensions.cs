using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace OptunaSharp.Visualization.Plotly.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the plotly visualization backend. Registers
///     the injectable <see cref="IOptunaPlotlyVisualizer" /> facade over the static
///     <see cref="PlotlyVisualization" />, carrying an optional app-wide default
///     <see cref="PlotlyChartCustomization" /> (house figure styling — size, template, static-export
///     config, …) that every figure the visualizer produces picks up automatically.
///     <para>
///     <b>Honest scope:</b> the visualization API is static, so this facade exists for DI ergonomics
///     (testability via a mockable interface, and one injected place to carry default styling), not
///     because rendering needs a service. Code that needs neither can keep calling
///     <see cref="PlotlyVisualization" /> directly.
///     </para>
/// </summary>
public static class PlotlyVisualizationServiceCollectionExtensions
{
    /// <summary>
    ///     Register the <see cref="IOptunaPlotlyVisualizer" /> facade (and the default
    ///     <see cref="PlotlyChartCustomization" />, when supplied). Idempotent — a second call is a no-op
    ///     (<c>TryAdd</c>), so the first registration's default customization wins.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="defaultCustomization">
    ///     The app-wide default figure styling applied to every figure the visualizer produces, or
    ///     <c>null</c> for canonical (unstyled) figures. When non-null it is also registered as a resolvable
    ///     <see cref="PlotlyChartCustomization" /> singleton.
    /// </param>
    /// <param name="lifetime">The registration lifetime for the visualizer; defaults to <see cref="ServiceLifetime.Singleton" /> (the facade is stateless and safe to share).</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaPlotlyVisualization(
        this IServiceCollection services,
        PlotlyChartCustomization? defaultCustomization = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);

        // Register the default customization (when supplied) so it is injectable in its own right; the
        // visualizer factory closes over the local value directly to keep behavior independent of any
        // other PlotlyChartCustomization registration.
        if (defaultCustomization is not null)
            services.TryAddSingleton(defaultCustomization);

        services.TryAdd(new ServiceDescriptor(
            typeof(IOptunaPlotlyVisualizer),
            _ => new OptunaPlotlyVisualizer(defaultCustomization),
            lifetime));
        return services;
    }
}
