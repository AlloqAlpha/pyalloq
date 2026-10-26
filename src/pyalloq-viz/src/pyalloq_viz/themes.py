from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt

ThemeName = Literal["alloq_dark", "alloq_light"]


@dataclass(frozen=True)
class ThemePalette:
    name: ThemeName
    bg_primary: str
    bg_secondary: str
    bg_card: str
    text_primary: str
    text_secondary: str
    grid_color: str
    border_color: str
    # Semantic quant colors
    emerald: str  # Positive returns / median trajectory / bull regime / prior
    teal: str  # Secondary positive / inner confidence cone
    cyan: str  # Outer confidence cone / benchmark
    amber: str  # Warning / VaR marker / neutral regime / target weights
    coral: str  # Tail risk / CVaR / drawdown median
    crimson: str  # Ruin threshold / extreme crash / bear regime
    violet: str  # Entropy posterior / alternative strategy / ML cluster
    indigo: str  # Secondary accent
    slate: str  # Muted paths / neutral bars
    categorical: tuple[str, ...]


ALLOQ_DARK = ThemePalette(
    name="alloq_dark",
    bg_primary="#0B0F19",
    bg_secondary="#111827",
    bg_card="#1E293B",
    text_primary="#F8FAFC",
    text_secondary="#94A3B8",
    grid_color="rgba(148, 163, 184, 0.14)",
    border_color="#334155",
    emerald="#10B981",
    teal="#14B8A6",
    cyan="#06B6D4",
    amber="#F59E0B",
    coral="#F43F5E",
    crimson="#EF4444",
    violet="#8B5CF6",
    indigo="#6366F1",
    slate="#64748B",
    categorical=(
        "#10B981",
        "#06B6D4",
        "#8B5CF6",
        "#F59E0B",
        "#F43F5E",
        "#3B82F6",
        "#EC4899",
        "#14B8A6",
        "#A855F7",
        "#EAB308",
    ),
)

ALLOQ_LIGHT = ThemePalette(
    name="alloq_light",
    bg_primary="#FFFFFF",
    bg_secondary="#F8FAFC",
    bg_card="#F1F5F9",
    text_primary="#0F172A",
    text_secondary="#475569",
    grid_color="rgba(71, 85, 105, 0.15)",
    border_color="#CBD5E1",
    emerald="#059669",
    teal="#0D9488",
    cyan="#0891B2",
    amber="#D97706",
    coral="#E11D48",
    crimson="#DC2626",
    violet="#7C3AED",
    indigo="#4F46E5",
    slate="#64748B",
    categorical=(
        "#059669",
        "#0891B2",
        "#7C3AED",
        "#D97706",
        "#E11D48",
        "#2563EB",
        "#DB2777",
        "#0D9488",
        "#9333EA",
        "#CA8A04",
    ),
)


def get_theme(theme: ThemeName | str = "alloq_dark") -> ThemePalette:
    """Returns the ThemePalette corresponding to the theme name."""
    if theme == "alloq_light":
        return ALLOQ_LIGHT
    return ALLOQ_DARK


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Converts a 6-digit hex string (e.g. '#10B981') into an 'rgba(r, g, b, a)' string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def apply_plotly_theme(
    fig: Any,
    theme: ThemeName | str = "alloq_dark",
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> Any:
    """
    Applies the institutional PyAlloq layout styling to a Plotly Figure or Matplotlib Figure.
    """
    palette = get_theme(theme)

    # Check if fig is a Plotly Figure (duck-typed via update_layout)
    if hasattr(fig, "update_layout"):
        title_text = title
        if title and subtitle:
            title_text = (
                f"<b>{title}</b><br>"
                f"<span style='font-size:12px;color:{palette.text_secondary}'>{subtitle}</span>"
            )
        elif title:
            title_text = f"<b>{title}</b>"

        layout_kwargs: dict[str, Any] = {
            "paper_bgcolor": palette.bg_primary,
            "plot_bgcolor": palette.bg_secondary,
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "color": palette.text_primary,
                "size": 12,
            },
            "colorway": list(palette.categorical),
            "margin": {"l": 60, "r": 40, "t": 80 if title_text else 45, "b": 55},
            "hovermode": "x unified",
            "legend": {
                "bgcolor": hex_to_rgba(
                    palette.bg_card if palette.name == "alloq_dark" else "#FFFFFF", 0.8
                ),
                "bordercolor": palette.border_color,
                "borderwidth": 1,
                "font": {"color": palette.text_primary, "size": 11},
            },
        }
        if title_text is not None:
            layout_kwargs["title"] = {
                "text": title_text,
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 17, "color": palette.text_primary},
            }
        if height is not None:
            layout_kwargs["height"] = height
        if width is not None:
            layout_kwargs["width"] = width

        fig.update_layout(**layout_kwargs)
        fig.update_xaxes(
            gridcolor=palette.grid_color,
            zerolinecolor=palette.border_color,
            linecolor=palette.border_color,
            tickfont={"color": palette.text_secondary},
            title_font={"color": palette.text_primary},
        )
        fig.update_yaxes(
            gridcolor=palette.grid_color,
            zerolinecolor=palette.border_color,
            linecolor=palette.border_color,
            tickfont={"color": palette.text_secondary},
            title_font={"color": palette.text_primary},
        )
        # Update subplot titles if present
        if hasattr(fig.layout, "annotations") and fig.layout.annotations:
            for ann in fig.layout.annotations:
                if hasattr(ann, "font") and ann.font is not None and not ann.font.color:
                    ann.font.color = palette.text_primary
        return fig

    # Matplotlib Figure fallback
    if isinstance(fig, plt.Figure):
        fig.patch.set_facecolor(palette.bg_primary)
        for ax in fig.axes:
            ax.set_facecolor(palette.bg_secondary)
            ax.tick_params(colors=palette.text_secondary)
            ax.xaxis.label.set_color(palette.text_primary)
            ax.yaxis.label.set_color(palette.text_primary)
            ax.title.set_color(palette.text_primary)
            for spine in ax.spines.values():
                spine.set_edgecolor(palette.border_color)
            ax.grid(True, color=palette.border_color, alpha=0.3, linestyle="--")
        if title:
            fig.suptitle(
                title, color=palette.text_primary, fontsize=14, fontweight="bold"
            )
        return fig

    return fig
