from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from pyalloq_viz.themes import ThemeName, get_theme


class BaseTearSheet(ABC):
    """
    Abstract base class for PyAlloq Visualization Tear Sheets.
    Provides:
      - `.to_figure(theme=...)`: Returns a combined multi-subplot `plotly.graph_objects.Figure`.
      - `.to_html(path=..., theme=...)`: Renders a self-contained HTML dashboard with KPI cards
        and interactive Plotly charts.
      - `.show(theme=...)`: Displays the combined tear sheet figure interactively.
      - `.save_image(path=..., ...)`: Exports the tear sheet to PNG/PDF/SVG via Kaleido.
    """

    title: str = "PyAlloq Tear Sheet"
    subtitle: str = "Quantitative Portfolio Analytics & Diagnostics"

    @abstractmethod
    def get_kpis(self) -> dict[str, str]:
        """Returns key performance and diagnostic indicators for the HTML report header."""

    @abstractmethod
    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        """Returns the list of individual Plotly figures that compose this tear sheet."""

    @abstractmethod
    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        """Returns a single unified multi-subplot Plotly Figure combining the tear sheet views."""

    def show(self, theme: ThemeName | str = "alloq_dark", **kwargs: Any) -> None:
        """Displays the unified multi-subplot tear sheet figure."""
        fig = self.to_figure(theme=theme)
        fig.show(**kwargs)

    def save_image(
        self,
        path: str | Path,
        theme: ThemeName | str = "alloq_dark",
        format: str | None = None,
        scale: int = 2,
        width: int = 1600,
        height: int = 1100,
    ) -> Path:
        """Exports the combined tear sheet figure to a static image file (requires kaleido)."""
        out_path = Path(path)
        fig = self.to_figure(theme=theme)
        fig.write_image(
            str(out_path), format=format, scale=scale, width=width, height=height
        )
        return out_path

    def to_html(
        self,
        path: str | Path | None = None,
        theme: ThemeName | str = "alloq_dark",
        title: str | None = None,
        include_plotlyjs: str | bool = "cdn",
    ) -> str:
        """
        Generates a standalone, responsive HTML tear sheet report containing KPI summary cards
        and all interactive component charts. Writes to `path` if provided and returns the HTML string.
        """
        palette = get_theme(theme)
        report_title = title or self.title
        kpis = self.get_kpis()
        figures = self.get_component_figures(theme=theme)

        kpi_cards_html = []
        for label, value in kpis.items():
            kpi_cards_html.append(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """
            )

        chart_blocks_html = []
        for idx, fig in enumerate(figures):
            js_mode = include_plotlyjs if idx == 0 else False
            fig_html = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=js_mode,
                config={"responsive": True, "displaylogo": False},
            )
            chart_blocks_html.append(f'<div class="chart-card">{fig_html}</div>')

        timestamp_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        :root {{
            --bg-primary: {palette.bg_primary};
            --bg-secondary: {palette.bg_secondary};
            --bg-card: {palette.bg_card};
            --text-primary: {palette.text_primary};
            --text-secondary: {palette.text_secondary};
            --border-color: {palette.border_color};
            --accent-emerald: {palette.emerald};
            --accent-cyan: {palette.cyan};
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 28px 36px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }}
        .header-title {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .header-subtitle {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .header-meta {{
            font-size: 12px;
            color: var(--text-secondary);
            text-align: right;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
            gap: 14px;
            margin-bottom: 26px;
        }}
        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px 18px;
        }}
        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .charts-container {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        .chart-card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <header class="header">
        <div>
            <div class="header-title">{report_title}</div>
            <div class="header-subtitle">{self.subtitle}</div>
        </div>
        <div class="header-meta">
            <div>Generated by <strong>pyalloq-viz</strong></div>
            <div>{timestamp_str}</div>
        </div>
    </header>
    <section class="kpi-grid">
        {"".join(kpi_cards_html)}
    </section>
    <main class="charts-container">
        {"".join(chart_blocks_html)}
    </main>
</body>
</html>"""

        if path is not None:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_doc, encoding="utf-8")

        return html_doc
