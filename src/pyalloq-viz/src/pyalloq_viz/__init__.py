from pyalloq_viz.allocation import (
    plot_correlation_and_periodic_returns,
    plot_hierarchical_composition,
    plot_multi_universe_frontier,
    plot_risk_contributions,
    plot_weight_drift_turnover,
)
from pyalloq_viz.comparison import (
    plot_attribution_waterfall,
    plot_equity_drawdown_overlay,
    plot_multi_strategy_fan_chart,
    plot_multi_turnover_drift,
)
from pyalloq_viz.contagion import (
    plot_fragility_hawkes_regime_portrait,
    plot_spectral_contagion_network,
)
from pyalloq_viz.multi_period import (
    plot_alpha_decay_surface,
    plot_mpc_trajectory,
    plot_mpo_pillar_comparison,
    plot_thermodynamic_regime_gate,
)
from pyalloq_viz.regime import plot_entropy_pooling, plot_regime_transitions
from pyalloq_viz.robustness import plot_cpcv_paths, plot_pbo_distribution
from pyalloq_viz.simulation import (
    plot_cvar_tail_risk,
    plot_drawdown_fan_chart,
    plot_fan_chart,
    plot_terminal_wealth_ruin,
)
from pyalloq_viz.tearsheets import (
    AllocationTearSheet,
    BaseTearSheet,
    ComparisonTearSheet,
    MacroRegimeTearSheet,
    MultiPeriodTearSheet,
    RobustnessTearSheet,
    SimulationTearSheet,
)
from pyalloq_viz.themes import (
    ALLOQ_DARK,
    ALLOQ_LIGHT,
    ThemeName,
    ThemePalette,
    apply_plotly_theme,
    get_theme,
    hex_to_rgba,
)
from pyalloq_viz.utils import (
    compute_drawdown_paths,
    compute_kde_1d,
    compute_risk_contributions,
    extract_equity_paths,
    reconstruct_drifted_weights,
    weighted_quantiles,
)

__all__ = [
    "ALLOQ_DARK",
    "ALLOQ_LIGHT",
    "AllocationTearSheet",
    # Base & Themes
    "BaseTearSheet",
    "ComparisonTearSheet",
    "MacroRegimeTearSheet",
    "MultiPeriodTearSheet",
    "RobustnessTearSheet",
    "SimulationTearSheet",
    "ThemeName",
    "ThemePalette",
    "apply_plotly_theme",
    "compute_drawdown_paths",
    "compute_kde_1d",
    "compute_risk_contributions",
    "extract_equity_paths",
    "get_theme",
    "hex_to_rgba",
    "plot_alpha_decay_surface",
    "plot_attribution_waterfall",
    "plot_correlation_and_periodic_returns",
    "plot_cpcv_paths",
    "plot_cvar_tail_risk",
    "plot_drawdown_fan_chart",
    # Macro Regime & Entropy
    "plot_entropy_pooling",
    "plot_equity_drawdown_overlay",
    # Simulation & Multiverse
    "plot_fan_chart",
    "plot_fragility_hawkes_regime_portrait",
    "plot_hierarchical_composition",
    "plot_mpc_trajectory",
    "plot_mpo_pillar_comparison",
    "plot_multi_strategy_fan_chart",
    "plot_multi_turnover_drift",
    # Allocation Mechanics & Risk
    "plot_multi_universe_frontier",
    # Robustness & Overfitting
    "plot_pbo_distribution",
    "plot_regime_transitions",
    "plot_risk_contributions",
    "plot_spectral_contagion_network",
    "plot_terminal_wealth_ruin",
    "plot_thermodynamic_regime_gate",
    "plot_weight_drift_turnover",
    "reconstruct_drifted_weights",
    # Utilities
    "weighted_quantiles",
]
