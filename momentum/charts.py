"""Altair chart builders for the UI (pure functions: data in, chart spec out).

Kept free of Streamlit so the charts can be rendered and inspected outside the
app. Colours come from a palette dict (``PALETTES["light"|"dark"]``) — the
reference data-viz palette: categorical hues in a fixed order (strategy = slot 1
blue, benchmark = slot 2 orange), blue↔red with a neutral midpoint for signed
values, and a single-hue blue ramp for magnitudes.
"""

from __future__ import annotations

import json
import re

import altair as alt
import numpy as np
import pandas as pd

from .analytics import MONTHS

PALETTES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec",
        "ink": "#0b0b0b", "muted": "#898781", "surface": "#fcfcfb",
        "seq": ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"],
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "pos": "#3987e5", "neg": "#e66767", "mid": "#383835",
        "ink": "#ffffff", "muted": "#898781", "surface": "#1a1a19",
        "seq": ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
    },
}


def _field(name: str) -> str:
    """Escape a column name for use as a Vega-Lite field (dots/brackets are paths)."""
    return re.sub(r"([.\[\]])", r"\\\1", name)


def strat_bench_colors(pal: dict, bench_label: str | None) -> dict[str, str]:
    cols = {"Strategy": pal["series"][0]}
    if bench_label:
        cols[bench_label] = pal["series"][1]
    return cols


def line_chart(wide: pd.DataFrame, colors: dict[str, str], y_title: str, pal: dict, y_fmt: str = ",.0f",
               height: int = 360, log: bool = False, dashed: tuple[str, ...] = ()) -> alt.LayerChart:
    """Multi-series line chart with a hover crosshair + all-series tooltip."""
    domain = [c for c in colors if c in wide.columns]
    wide = wide[domain]
    long = (wide.rename_axis("Date").reset_index()
            .melt("Date", var_name="Series", value_name="Value").dropna())
    color = alt.Color("Series:N", scale=alt.Scale(domain=domain, range=[colors[c] for c in domain]),
                      legend=alt.Legend(orient="top", title=None))
    y = alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=y_fmt),
              scale=alt.Scale(type="log") if log else alt.Scale(zero=False))
    span_years = (wide.index.max() - wide.index.min()).days / 365.25 if len(wide) else 0
    # Multi-year charts: one tick per year (Vega's default adds Apr/Jul/Oct and clutters).
    x_axis = alt.Axis(format="%Y", tickCount={"interval": "year", "step": 1}) if span_years >= 3 else alt.Axis()
    base = alt.Chart(long).encode(x=alt.X("Date:T", title=None, axis=x_axis), y=y, color=color)
    lines = base.mark_line(strokeWidth=2).encode(
        strokeDash=alt.condition(alt.FieldOneOfPredicate(field="Series", oneOf=list(dashed) or ["__none__"]),
                                 alt.value([4, 3]), alt.value([1, 0])))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False)
    tips = [alt.Tooltip(field="Date", type="temporal", title="Date")] + [
        alt.Tooltip(field=_field(c), type="quantitative", title=c, format=y_fmt) for c in domain]
    rule = (alt.Chart(long).transform_pivot("Series", value="Value", groupby=["Date"])
            .mark_rule(color=pal["muted"], strokeWidth=1)
            .encode(x="Date:T", opacity=alt.condition(nearest, alt.value(0.7), alt.value(0)), tooltip=tips)
            .add_params(nearest))
    points = base.mark_point(size=64, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    return alt.layer(lines, points, rule).properties(height=height)


def diverging_bars(df: pd.DataFrame, cat: str, val: str, fmt: str, title: str, pal: dict,
                   step: int = 24) -> alt.Chart:
    """Horizontal bars coloured by sign (blue ↑ / red ↓), sorted by value.

    Sized per bar (``step`` px) rather than by total height: Streamlit fits a
    fixed height around axes too, which squeezed bands and dropped labels.
    """
    color = alt.condition(alt.datum[val] >= 0, alt.value(pal["pos"]), alt.value(pal["neg"]))
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
            .encode(y=alt.Y(f"{cat}:N", sort=alt.EncodingSortField(field=val, order="descending"), title=None),
                    x=alt.X(f"{val}:Q", title=title, axis=alt.Axis(format=fmt)),
                    color=color,
                    tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt)])
            .properties(height=alt.Step(step)))


def calendar_heatmap(table: pd.DataFrame, pal: dict) -> alt.LayerChart:
    """Year × month return grid: blue gains, red losses, grey ≈ 0, labelled cells."""
    long = (table[MONTHS].reset_index()
            .melt("Year", var_name="Month", value_name="ret").dropna())
    lim = float(max(0.03, long["ret"].abs().quantile(0.95))) if len(long) else 0.05
    x = alt.X("Month:N", sort=MONTHS, title=None, axis=alt.Axis(orient="top", labelAngle=0))
    y = alt.Y("Year:O", title=None)
    rect = (alt.Chart(long).mark_rect(stroke=pal["surface"], strokeWidth=2, cornerRadius=3)
            .encode(x=x, y=y,
                    color=alt.Color("ret:Q", title="Monthly return",
                                    # Lab interpolation keeps each arm on its own hue
                                    # (HCL would swing the gain arm through pink/purple).
                                    scale=alt.Scale(domain=[-lim, 0, lim], interpolate="lab",
                                                    range=[pal["neg"], pal["mid"], pal["pos"]], clamp=True),
                                    legend=alt.Legend(format="%", orient="bottom")),
                    tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Month:N"),
                             alt.Tooltip("ret:Q", title="Return", format=".2%")]))
    text = (alt.Chart(long).mark_text(fontSize=11)
            .encode(x=x, y=y, text=alt.Text("ret:Q", format=".1%"),
                    color=alt.condition(f"abs(datum.ret) > {lim * 0.55}", alt.value("#ffffff"),
                                        alt.value(pal["ink"]))))
    return alt.layer(rect, text).properties(height=alt.Step(30))


def sector_bar(sec: pd.DataFrame, pal: dict) -> alt.Chart:
    """Scanner sector mix: one bar per industry (weight), names count in the tooltip."""
    return (alt.Chart(sec).mark_bar(cornerRadiusEnd=4, color=pal["series"][0])
            .encode(y=alt.Y("Industry:N", sort="-x", title=None),
                    x=alt.X("weight:Q", title="Weight", axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("Industry:N"), alt.Tooltip("weight:Q", format=".1%"),
                             alt.Tooltip("names:Q", title="Names")])
            .properties(height=alt.Step(26)))


def weights_bar(df: pd.DataFrame, pal: dict, label: str = "Symbol", value: str = "Weight") -> alt.Chart:
    """Portfolio weights, largest first (``value`` in percent units, e.g. 5.0 = 5%)."""
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=pal["series"][0])
            .encode(x=alt.X(f"{label}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-60)),
                    y=alt.Y(f"{value}:Q", title="Weight (%)"),
                    tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=".2f", title="Weight %")])
            .properties(height=320))


def sector_area(sx: pd.DataFrame, pal: dict, top: int = 7, max_points: int = 300) -> alt.Chart:
    """Stacked industry weights over time: the ``top`` industries + grey 'Other'."""
    top_ind = list(sx.columns[:top])
    plot = sx[top_ind].copy()
    if sx.shape[1] > top:
        plot["Other"] = sx[sx.columns[top:]].sum(axis=1)
    if len(plot) > max_points:  # keep the chart light for daily/weekly rebalancing
        plot = plot.iloc[:: int(np.ceil(len(plot) / max_points))]
    long = plot.rename_axis("Date").reset_index().melt("Date", var_name="Industry", value_name="Weight")
    domain = list(plot.columns)
    rng = pal["series"][: len(top_ind)] + ([pal["muted"]] if "Other" in domain else [])
    return (alt.Chart(long).mark_area(interpolate="step-after", stroke=pal["surface"], strokeWidth=1)
            .encode(x=alt.X("Date:T", title=None),
                    y=alt.Y("Weight:Q", stack="zero", axis=alt.Axis(format="%"), title="Portfolio weight",
                            scale=alt.Scale(domain=[0, 1], nice=False)),
                    color=alt.Color("Industry:N", scale=alt.Scale(domain=domain, range=rng),
                                    sort=domain, legend=alt.Legend(orient="top", title=None, columns=4)),
                    order=alt.Order("order:Q"),
                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Industry:N"),
                             alt.Tooltip("Weight:Q", format=".1%")])
            .transform_calculate(order=f"indexof({json.dumps(domain)}, datum.Industry)")
            .properties(height=300))


def metric_heatmap(table: pd.DataFrame, metric: str, order: list[str], pal: dict, fmt: str,
                   lower_is_better: bool, tooltip_metrics: dict[str, str]) -> alt.LayerChart:
    """Frequency × Top-N grid coloured by ``metric`` (darker = better), labelled cells.

    ``tooltip_metrics`` maps metric column -> d3 format for the hover card.
    """
    seq = pal["seq"][::-1] if lower_is_better else pal["seq"]
    heat = alt.Chart(table).encode(
        x=alt.X("top_n:O", title="Top N", axis=alt.Axis(labelAngle=0)), y=alt.Y("frequency:N", sort=order, title=None))
    rect = heat.mark_rect(stroke=pal["surface"], strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(f"{metric}:Q", scale=alt.Scale(range=seq, interpolate="lab"),
                        legend=alt.Legend(format=fmt, title=None)),
        tooltip=[alt.Tooltip("label:N", title="Combination")] + [
            alt.Tooltip(f"{m}:Q", format=f) for m, f in tooltip_metrics.items()])
    vals = table[metric].astype(float)
    lo, hi = (float(vals.min()), float(vals.max())) if vals.notna().any() else (0.0, 0.0)
    # Light text on the dark half of the ramp (dark = better = high, or low when lower is better).
    cut = lo + (hi - lo) * (0.45 if lower_is_better else 0.55)
    op = "<" if lower_is_better else ">"
    text = heat.mark_text(fontSize=13).encode(
        text=alt.Text(f"{metric}:Q", format=fmt),
        color=alt.condition(f"datum.{metric} {op} {cut}", alt.value("#ffffff"), alt.value(pal["ink"])))
    return alt.layer(rect, text).properties(height=alt.Step(48))
