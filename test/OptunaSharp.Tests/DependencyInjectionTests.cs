using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Extensions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Visualization.Plotly;
using OptunaSharp.Visualization.Plotly.Extensions;

namespace OptunaSharp.Tests;

/// <summary>
///     Tests for the Microsoft.Extensions.DependencyInjection wire-up: the Core
///     <c>OptunaSharp.Extensions</c> composition root (sampler / pruner / storage + study factory) and the
///     Visualization.Plotly <see cref="IOptunaPlotlyVisualizer" /> facade. These prove the registrations
///     resolve the expected instances and that the study factory honors both the container's services and
///     Optuna's own defaults when a piece is left unregistered.
/// </summary>
[TestClass]
public class DependencyInjectionTests
{
    [TestMethod]
    public void AddOptunaStudyFactory_Composes_RegisteredSamplerStorageAndDefaultDirection()
    {
        var sampler = new RandomSampler(seed: 0);
        var storage = new InMemoryStorage();

        var provider = new ServiceCollection()
            .AddOptunaSampler(sampler)
            .AddOptunaStorage(storage)
            .AddOptunaStudyFactory(o => o.DefaultDirection = StudyDirection.Maximize)
            .BuildServiceProvider();

        // The abstractions resolve to the exact instances registered.
        provider.GetRequiredService<BaseSampler>().Should().BeSameAs(sampler);
        provider.GetRequiredService<BaseStorage>().Should().BeSameAs(storage);

        var factory = provider.GetRequiredService<IOptunaStudyFactory>();
        Study study = factory.CreateStudy();

        // The option default is applied when CreateStudy is called with no explicit direction.
        study.Direction.Should().Be(StudyDirection.Maximize);

        study.Optimize(t => t.SuggestFloat("x", -10, 10), nTrials: 5);
        study.Trials.Should().HaveCount(5);
    }

    [TestMethod]
    public void CreateStudy_ExplicitArguments_OverrideOptionDefaults()
    {
        var provider = new ServiceCollection()
            .AddOptunaStorage(new InMemoryStorage())
            .AddOptunaStudyFactory(o => o.DefaultDirection = StudyDirection.Maximize)
            .BuildServiceProvider();

        var factory = provider.GetRequiredService<IOptunaStudyFactory>();

        // An explicit direction argument wins over OptunaStudyOptions.DefaultDirection.
        Study study = factory.CreateStudy(direction: StudyDirection.Minimize);
        study.Direction.Should().Be(StudyDirection.Minimize);
    }

    [TestMethod]
    public void AddOptunaStudyFactory_WithoutRegistrations_UsesOptunaDefaults()
    {
        // No sampler / pruner / storage registered: the factory must fall back to Optuna's own defaults
        // (in-memory storage, single-objective minimize) rather than throwing.
        var provider = new ServiceCollection()
            .AddOptunaStudyFactory()
            .BuildServiceProvider();

        Study study = provider.GetRequiredService<IOptunaStudyFactory>().CreateStudy();

        study.Should().NotBeNull();
        study.Direction.Should().Be(StudyDirection.Minimize);

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3);
        study.Trials.Should().HaveCount(3);
    }

    [TestMethod]
    public void AddOptunaSampler_Factory_HonorsTransientLifetime()
    {
        var provider = new ServiceCollection()
            .AddOptunaSampler(_ => new RandomSampler(seed: 1), ServiceLifetime.Transient)
            .BuildServiceProvider();

        // A transient registration yields a fresh sampler per resolution.
        var first = provider.GetRequiredService<BaseSampler>();
        var second = provider.GetRequiredService<BaseSampler>();
        first.Should().NotBeSameAs(second);
    }

    /// <summary>
    ///     Regression for the shared-sampler default: the sampler and pruner factory registrations default to
    ///     transient, so two studies built by the study factory never share a sampler or a pruner. Both hold
    ///     per-study state, and a Python <c>create_study()</c> builds its own default sampler per call.
    /// </summary>
    [TestMethod]
    public void StudyFactory_DefaultFactoryRegistrations_GiveEachStudyItsOwnSamplerAndPruner()
    {
        var services = new ServiceCollection()
            .AddOptunaSampler(_ => new RandomSampler(seed: 1))
            .AddOptunaPruner(_ => new MedianPruner())
            .AddOptunaStudyFactory();

        services.Single(d => d.ServiceType == typeof(BaseSampler)).Lifetime.Should().Be(ServiceLifetime.Transient);
        services.Single(d => d.ServiceType == typeof(BasePruner)).Lifetime.Should().Be(ServiceLifetime.Transient);

        using var provider = services.BuildServiceProvider(
            new ServiceProviderOptions { ValidateScopes = true, ValidateOnBuild = true });
        var factory = provider.GetRequiredService<IOptunaStudyFactory>();
        Study first = factory.CreateStudy();
        Study second = factory.CreateStudy();

        first.Sampler.Should().NotBeSameAs(second.Sampler);
        first.Pruner.Should().NotBeSameAs(second.Pruner);
    }

    /// <summary>
    ///     Regression for the root-provider factory: resolved inside a scope, the study factory takes a scoped
    ///     sampler and storage from that scope. The singleton factory resolved them from the root, which
    ///     scope validation rejects ("Cannot resolve scoped service … from root provider") and which, without
    ///     validation, turned the scoped registrations into application-wide singletons.
    /// </summary>
    [TestMethod]
    public void StudyFactory_ResolvedInScope_UsesThatScopesRegistrations()
    {
        var services = new ServiceCollection()
            .AddOptunaSampler(_ => new RandomSampler(seed: 2), ServiceLifetime.Scoped)
            .AddOptunaStorage(_ => new InMemoryStorage(), ServiceLifetime.Scoped)
            .AddOptunaStudyFactory();
        services.Single(d => d.ServiceType == typeof(IOptunaStudyFactory)).Lifetime
            .Should().Be(ServiceLifetime.Transient);

        using var provider = services.BuildServiceProvider(
            new ServiceProviderOptions { ValidateScopes = true, ValidateOnBuild = true });
        using var scope = provider.CreateScope();
        Study study = scope.ServiceProvider.GetRequiredService<IOptunaStudyFactory>().CreateStudy(studyName: "scoped");

        study.Sampler.Should().BeSameAs(scope.ServiceProvider.GetRequiredService<BaseSampler>());
        scope.ServiceProvider.GetRequiredService<BaseStorage>().GetAllStudies()
            .Should().ContainSingle(s => s.StudyName == "scoped");

        // Another scope has its own storage, which the first scope's study never touched.
        using var otherScope = provider.CreateScope();
        otherScope.ServiceProvider.GetRequiredService<BaseStorage>().GetAllStudies().Should().BeEmpty();
    }

    [TestMethod]
    public void AddOptunaPlotlyVisualization_Registers_FacadeAndDefaultCustomization()
    {
        var customization = new PlotlyChartCustomization();

        var provider = new ServiceCollection()
            .AddOptunaPlotlyVisualization(customization)
            .BuildServiceProvider();

        var visualizer = provider.GetRequiredService<IOptunaPlotlyVisualizer>();
        visualizer.DefaultCustomization.Should().BeSameAs(customization);

        // The default customization is separately resolvable.
        provider.GetRequiredService<PlotlyChartCustomization>().Should().BeSameAs(customization);

        // The facade produces a real figure over a completed study.
        var study = Optuna.CreateStudy();
        study.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 4);
        visualizer.PlotOptimizationHistory(study).Should().NotBeNull();
    }

    [TestMethod]
    public void AddOptunaPlotlyVisualization_WithoutCustomization_ResolvesFacade()
    {
        var provider = new ServiceCollection()
            .AddOptunaPlotlyVisualization()
            .BuildServiceProvider();

        var visualizer = provider.GetRequiredService<IOptunaPlotlyVisualizer>();
        visualizer.DefaultCustomization.Should().BeNull();
    }
}
