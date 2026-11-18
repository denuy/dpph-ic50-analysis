"""DPPH radical-scavenging analysis: percent scavenging and IC50 from replicate absorbances.

The DPPH assay measures how well an extract or compound neutralises the stable
DPPH radical. Absorbance at 517 nm falls as the radical is scavenged, so for each
extract concentration::

    % scavenging = (A_control - A_sample) / A_control * 100

where ``A_control`` is DPPH with no extract. The IC50 is the extract concentration
that scavenges 50 % of the radical. It is estimated by fitting a logistic
(Hill / four-parameter) dose-response curve::

    response = bottom + (top - bottom) / (1 + (IC50 / x) ** hill)

By default ``bottom = 0`` and ``top = 100`` are fixed, which is the textbook
definition of the DPPH IC50 (the concentration giving exactly 50 % scavenging).
If ``bottom`` or ``top`` is freed, the fitted ``ic50`` is the curve's midpoint
(a "relative" IC50), which equals 50 % scavenging only when the plateaus are 0 and 100.

Command line::

    python dpph_ic50.py data/example_data.csv --plot fit.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "Analysis",
    "FitResult",
    "Readings",
    "analyze",
    "fit_ic50",
    "four_pl",
    "load_readings",
    "main",
    "percent_scavenging",
    "plot_analysis",
]

TARGET_PERCENT = 50.0


# --------------------------------------------------------------------------- #
# Core calculations
# --------------------------------------------------------------------------- #
def percent_scavenging(a_control, a_sample):
    """Percent DPPH scavenging: ``(A_control - A_sample) / A_control * 100``.

    ``a_control`` is a positive scalar (usually the mean of the control replicates);
    ``a_sample`` may be a scalar or any array. Values are not clipped, so noise can
    give slightly negative or >100 results.
    """
    a_control = np.asarray(a_control, dtype=float)
    if np.any(a_control <= 0):
        raise ValueError("control absorbance must be positive (check the blank correction)")
    return (a_control - np.asarray(a_sample, dtype=float)) / a_control * 100.0


def four_pl(x, bottom, top, ic50, hill):
    """Four-parameter logistic (Hill) curve; equals the midpoint of bottom and top at ``x = ic50``."""
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (1.0 + (ic50 / x) ** hill)


@dataclass(frozen=True)
class FitResult:
    """Outcome of an IC50 fit.

    ``status`` is one of:

    * ``"ok"`` - curve fitted; ``ic50`` and friends are set.
    * ``"not_reached"`` - no concentration reached 50 %; the IC50 is above the tested range.
    * ``"below_range"`` - even the lowest concentration was at or above 50 %; IC50 is below the range.
    * ``"fit_failed"`` - the optimiser did not converge.

    Only ``"ok"`` carries numbers; ``message`` explains every other status.
    """

    status: str
    message: str
    n_points: int
    ic50: Optional[float] = None
    ic50_se: Optional[float] = None
    ic50_ci95: Optional[tuple] = None
    hill: Optional[float] = None
    bottom: Optional[float] = None
    top: Optional[float] = None
    r_squared: Optional[float] = None
    dof: Optional[int] = None


def _initial_log10_ic50(x_unique: np.ndarray, mean_y: np.ndarray) -> float:
    """Starting guess: interpolate (in log concentration) where the data first cross 50 %."""
    log_x = np.log10(x_unique)
    for i in range(len(x_unique) - 1):
        lo, hi = mean_y[i], mean_y[i + 1]
        if lo != hi and (lo - TARGET_PERCENT) * (hi - TARGET_PERCENT) <= 0:
            frac = (TARGET_PERCENT - lo) / (hi - lo)
            return float(log_x[i] + frac * (log_x[i + 1] - log_x[i]))
    return float(np.mean(log_x))


def fit_ic50(x, y, bottom: Optional[float] = 0.0, top: Optional[float] = 100.0) -> FitResult:
    """Fit a Hill curve to percent-scavenging data and return the IC50 with its uncertainty.

    ``x`` are sample concentrations (> 0; repeats allowed, e.g. one entry per replicate)
    and ``y`` the matching percent scavenging. ``bottom`` / ``top`` are fixed at the given
    value, or fitted if ``None``. The IC50 is fitted as ``log10(IC50)`` so it stays positive;
    its standard error comes from the fit covariance (delta method) and the 95 % confidence
    interval from a t-distribution with ``n_points - n_free_parameters`` degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite")
    if np.any(x <= 0):
        raise ValueError("concentrations must be > 0 (the 0 row is the control, not a data point)")

    n_free = 2 + (bottom is None) + (top is None)
    x_unique = np.unique(x)
    if x_unique.size < n_free + 1:
        raise ValueError(
            f"need at least {n_free + 1} distinct concentrations to fit {n_free} parameters, "
            f"got {x_unique.size}"
        )
    mean_y = np.array([y[x == u].mean() for u in x_unique])

    # Edge cases: the 50 % level is not bracketed by the data, so an IC50 would be extrapolation.
    if mean_y.max() < TARGET_PERCENT:
        return FitResult(
            status="not_reached",
            message=(
                f"IC50 > {x_unique.max():g}: no concentration reached {TARGET_PERCENT:g}% "
                f"scavenging (highest mean {mean_y.max():.1f}%)"
            ),
            n_points=int(x.size),
        )
    if mean_y.min() >= TARGET_PERCENT:
        return FitResult(
            status="below_range",
            message=(
                f"IC50 < {x_unique.min():g}: the lowest concentration already scavenged "
                f"{mean_y.min():.1f}%; test lower concentrations"
            ),
            n_points=int(x.size),
        )

    def model(xv, *p):
        it = iter(p)
        log_ic50 = next(it)
        hill = next(it)
        b = bottom if bottom is not None else next(it)
        t = top if top is not None else next(it)
        return four_pl(xv, b, t, 10.0**log_ic50, hill)

    p0 = [_initial_log10_ic50(x_unique, mean_y), 1.0]
    lower = [math.log10(x_unique.min()) - 2.0, 0.05]
    upper = [math.log10(x_unique.max()) + 2.0, 20.0]
    if bottom is None:
        p0.append(float(np.clip(mean_y.min(), -49.0, 49.0)))
        lower.append(-50.0)
        upper.append(50.0)
    if top is None:
        p0.append(float(np.clip(mean_y.max(), 51.0, 149.0)))
        lower.append(50.0)
        upper.append(150.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            popt, pcov = optimize.curve_fit(
                model, x, y, p0=p0, bounds=(lower, upper), max_nfev=10000
            )
    except (RuntimeError, ValueError) as exc:
        return FitResult(status="fit_failed", message=f"curve fit failed: {exc}", n_points=int(x.size))

    it = iter(popt[2:])
    fit_bottom = bottom if bottom is not None else float(next(it))
    fit_top = top if top is not None else float(next(it))
    log_ic50, hill = float(popt[0]), float(popt[1])
    ic50 = 10.0**log_ic50

    dof = int(x.size - n_free)
    var_log = pcov[0, 0]
    if dof > 0 and np.isfinite(var_log) and var_log >= 0:
        se_log = math.sqrt(var_log)
        ic50_se = ic50 * math.log(10.0) * se_log
        t_crit = float(stats.t.ppf(0.975, dof))
        ci = (10.0 ** (log_ic50 - t_crit * se_log), 10.0 ** (log_ic50 + t_crit * se_log))
    else:
        ic50_se, ci = None, None

    resid = y - model(x, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else None

    return FitResult(
        status="ok",
        message="",
        n_points=int(x.size),
        ic50=ic50,
        ic50_se=ic50_se,
        ic50_ci95=ci,
        hill=hill,
        bottom=float(fit_bottom),
        top=float(fit_top),
        r_squared=r_squared,
        dof=dof,
    )


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Readings:
    """Raw absorbances for one sample.

    ``concentrations`` are strictly increasing and > 0, ``absorbances`` has shape
    ``(n_concentrations, n_replicates)`` and ``control`` holds the DPPH-only replicates.
    """

    concentrations: np.ndarray
    absorbances: np.ndarray
    control: np.ndarray

    def __post_init__(self):
        conc = np.asarray(self.concentrations, dtype=float)
        absorb = np.asarray(self.absorbances, dtype=float)
        control = np.asarray(self.control, dtype=float)
        if conc.ndim != 1 or absorb.ndim != 2 or control.ndim != 1 or control.size == 0:
            raise ValueError("expected 1-D concentrations, 2-D absorbances and 1-D control")
        if absorb.shape[0] != conc.size:
            raise ValueError("absorbances needs one row per concentration")
        if np.any(conc <= 0) or np.any(np.diff(conc) <= 0):
            raise ValueError("concentrations must be > 0, strictly increasing and unique")
        object.__setattr__(self, "concentrations", conc)
        object.__setattr__(self, "absorbances", absorb)
        object.__setattr__(self, "control", control)


@dataclass(frozen=True)
class Analysis:
    """Per-concentration percent scavenging plus the IC50 fit."""

    concentrations: np.ndarray
    replicate_percent: np.ndarray  # (n_concentrations, n_replicates)
    mean_percent: np.ndarray
    sd_percent: np.ndarray  # sample SD (ddof=1); NaN with a single replicate
    fit: FitResult


def load_readings(path) -> Readings:
    """Read a CSV of absorbances.

    Layout: a header row, then one row per concentration - concentration first, then one
    column per replicate. The row with concentration ``0`` is the DPPH control. Blank lines
    and lines starting with ``#`` are ignored; rows may be in any order.
    """
    control = None
    rows = {}
    with open(path, newline="") as handle:
        lines = (ln for ln in handle if ln.strip() and not ln.lstrip().startswith("#"))
        header_seen = False
        for row in csv.reader(lines):
            try:
                conc = float(row[0])
            except ValueError:
                if header_seen:
                    raise ValueError(f"{path}: non-numeric concentration {row[0]!r}") from None
                header_seen = True  # first non-numeric row is the header
                continue
            header_seen = True
            values = [float(c) for c in row[1:] if c.strip() != ""]
            if not values:
                raise ValueError(f"{path}: no absorbance values for concentration {row[0]}")
            if conc == 0:
                if control is not None:
                    raise ValueError(f"{path}: more than one control (concentration 0) row")
                control = values
            else:
                if conc in rows:
                    raise ValueError(f"{path}: duplicate concentration {conc:g}")
                rows[conc] = values

    if control is None:
        raise ValueError(f"{path}: no control row (a row whose concentration is 0)")
    if not rows:
        raise ValueError(f"{path}: no sample rows")
    n_reps = {len(v) for v in rows.values()}
    if len(n_reps) != 1:
        raise ValueError(f"{path}: every concentration needs the same number of replicates")

    concs = sorted(rows)
    return Readings(
        concentrations=np.array(concs),
        absorbances=np.array([rows[c] for c in concs]),
        control=np.array(control),
    )


def analyze(
    readings: Readings,
    blank: float = 0.0,
    bottom: Optional[float] = 0.0,
    top: Optional[float] = 100.0,
    fit_on: str = "mean",
) -> Analysis:
    """Convert absorbances to percent scavenging and fit the IC50.

    ``blank`` (reagent blank absorbance, e.g. solvent without DPPH) is subtracted from every
    reading. The control is averaged to a single reference value. ``fit_on`` selects whether the
    curve is fitted to per-concentration means (``"mean"``) or to every replicate as its own
    point (``"replicates"``).
    """
    if fit_on not in ("mean", "replicates"):
        raise ValueError("fit_on must be 'mean' or 'replicates'")

    control_mean = float(np.mean(readings.control - blank))
    percent = percent_scavenging(control_mean, readings.absorbances - blank)
    mean = percent.mean(axis=1)
    n_reps = percent.shape[1]
    sd = percent.std(axis=1, ddof=1) if n_reps > 1 else np.full(mean.shape, np.nan)

    conc = readings.concentrations
    if fit_on == "mean":
        fit = fit_ic50(conc, mean, bottom=bottom, top=top)
    else:
        fit = fit_ic50(np.repeat(conc, n_reps), percent.ravel(), bottom=bottom, top=top)
    return Analysis(conc, percent, mean, sd, fit)


# --------------------------------------------------------------------------- #
# Plot and command line
# --------------------------------------------------------------------------- #
def plot_analysis(analysis: Analysis, path, unit: str = "ug/mL") -> None:
    """Save a dose-response plot: mean +/- SD points, the fitted curve and the IC50 marker."""
    from matplotlib.figure import Figure

    surface, ink, ink_2, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fit, conc = analysis.fit, analysis.concentrations
    n_reps = analysis.replicate_percent.shape[1]

    fig = Figure(figsize=(6.4, 4.2), dpi=200, facecolor=surface)
    ax = fig.add_subplot(1, 1, 1, facecolor=surface)
    ax.set_xscale("log")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink_2, labelsize=9)

    if fit.status == "ok":
        xs = np.logspace(np.log10(conc.min()) - 0.15, np.log10(conc.max()) + 0.15, 200)
        ax.plot(
            xs, four_pl(xs, fit.bottom, fit.top, fit.ic50, fit.hill),
            color=blue, linewidth=1.6, zorder=2, label="Hill fit",
        )
        ax.plot([conc.min() / 1.4, fit.ic50], [TARGET_PERCENT] * 2,
                color=ink_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.plot([fit.ic50] * 2, [-100, TARGET_PERCENT],
                color=ink_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        label = f"IC50 = {fit.ic50:.1f}"
        if fit.ic50_se is not None:
            label += f" ± {fit.ic50_se:.1f}"
        ax.annotate(
            f"{label} {unit}", xy=(fit.ic50, TARGET_PERCENT), xytext=(8, -16),
            textcoords="offset points", color=ink, fontsize=9, ha="left", va="top",
        )

    sd = np.nan_to_num(analysis.sd_percent)
    ax.errorbar(
        conc, analysis.mean_percent, yerr=sd, fmt="o", color=blue, ecolor=blue,
        elinewidth=1.0, capsize=3, markersize=7, markeredgecolor=surface,
        markeredgewidth=1.5, zorder=3, label=f"Mean ± SD (n = {n_reps})",
    )

    y_lo = min(0.0, float(np.min(analysis.mean_percent - sd))) - 5
    y_hi = max(100.0, float(np.max(analysis.mean_percent + sd))) + 5
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(conc.min() / 1.4, conc.max() * 1.4)
    ax.set_xticks(conc)  # label the tested concentrations themselves, not just powers of ten
    ax.set_xticklabels([f"{c:g}" for c in conc])
    ax.minorticks_off()
    ax.set_xlabel(f"Concentration ({unit})", color=ink_2, fontsize=10)
    ax.set_ylabel("DPPH scavenging (%)", color=ink_2, fontsize=10)
    title = "DPPH radical scavenging"
    if fit.status != "ok":
        title += f" – {fit.message}"
    ax.set_title(title, loc="left", color=ink, fontsize=11, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def _format_report(analysis: Analysis, unit: str) -> str:
    lines = [f"{'conc (' + unit + ')':>16} {'mean %':>9} {'SD %':>8}"]
    for c, m, s in zip(analysis.concentrations, analysis.mean_percent, analysis.sd_percent):
        lines.append(f"{c:>16g} {m:>9.2f} {s:>8.2f}")
    fit = analysis.fit
    lines.append("")
    if fit.status != "ok":
        lines.append(f"IC50: not estimated - {fit.message}")
        return "\n".join(lines)

    text = f"IC50 = {fit.ic50:.2f}"
    if fit.ic50_se is not None:
        text += f" +/- {fit.ic50_se:.2f} (SE)"
    text += f" {unit}"
    lines.append(text)
    if fit.ic50_ci95 is not None:
        lines.append(f"95% CI: {fit.ic50_ci95[0]:.2f} to {fit.ic50_ci95[1]:.2f} {unit}")
    detail = f"Hill slope = {fit.hill:.2f}"
    if fit.r_squared is not None:
        detail += f", R^2 = {fit.r_squared:.4f}"
    lines.append(f"{detail}, bottom = {fit.bottom:.1f}, top = {fit.top:.1f} (n = {fit.n_points} points)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DPPH percent scavenging and IC50 from replicate absorbances."
    )
    parser.add_argument("csv", type=Path, help="CSV of absorbances (see README for the layout)")
    parser.add_argument("--blank", type=float, default=0.0,
                        help="blank absorbance subtracted from every reading (default: 0)")
    parser.add_argument("--free-bottom", action="store_true", help="fit the bottom plateau instead of fixing it at 0")
    parser.add_argument("--free-top", action="store_true", help="fit the top plateau instead of fixing it at 100")
    parser.add_argument("--fit-on", choices=("mean", "replicates"), default="mean",
                        help="fit the replicate means (default) or every replicate")
    parser.add_argument("--unit", default="ug/mL", help="concentration unit for labels (default: ug/mL)")
    parser.add_argument("--plot", type=Path, metavar="PNG", help="save a dose-response plot to this file")
    args = parser.parse_args(argv)

    try:
        readings = load_readings(args.csv)
        analysis = analyze(
            readings,
            blank=args.blank,
            bottom=None if args.free_bottom else 0.0,
            top=None if args.free_top else 100.0,
            fit_on=args.fit_on,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_format_report(analysis, args.unit))
    if args.plot:
        plot_analysis(analysis, args.plot, unit=args.unit)
        print(f"plot saved to {args.plot}")
    return 1 if analysis.fit.status == "fit_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
