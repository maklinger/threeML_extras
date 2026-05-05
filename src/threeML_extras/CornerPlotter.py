"""
corner_plotter.py
-----------------
Standalone corner plot utility for UltraNest + 3ML results.

Colour palette: Paul Tol high-contrast (colorblind-safe)
    _CMAP_COLOR   #004488  blue  — contour fill cmap anchor
    _ACCENT_COLOR #BB5566  red   — median lines, ETI spans
    _LINE_COLOR   #004488  blue  — contour lines
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from corner import corner, overplot_lines, overplot_points


# ---------------------------------------------------------------------------
# Paul Tol high-contrast palette
# ---------------------------------------------------------------------------
_CMAP_COLOR   = "#004488"   # blue  — anchor for Blues-style cmap
_ACCENT_COLOR = "#BB5566"   # red   — median lines, ETI spans
_LINE_COLOR   = "#004488"   # blue  — contour lines
_GREY_CMAP    = "#BEBEBE"   # light grey


def _make_cmap(base_color: str = _CMAP_COLOR):
    """Blue-anchored sequential cmap with grey extremes."""
    cmap = LinearSegmentedColormap.from_list(
        "tol_blue", ["white", base_color], N=256
    )
    cmap.set_extremes(under=_GREY_CMAP, over=_GREY_CMAP, bad=_GREY_CMAP)
    return cmap


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def trim_name(name: str) -> str:
    """
    Derive a short label from a full 3ML parameter name.
    Falls back to the last dot-separated component with underscores as spaces.
    """
    short = name.split(".")[-1]
    if "cons" not in name:
        if "diff" in name:
            return short.split("_")[1]
        else:
            return " ".join(short.split("_")[:-1])
    else:
        return " ".join(short.split("_")[:2])


def _resolve_labels(paramnames: list, labels) -> list:
    """
    Resolve the labels argument into a list of strings.

    Parameters
    ----------
    paramnames : list of str
        Full 3ML parameter names from ur['paramnames'].
    labels : None | list | dict
        - None        → apply trim_name to each paramname
        - list of str → use directly (must match length of paramnames)
        - dict        → map from short name (last component) or full name
                        to label string; unmapped names fall back to trim_name
    """
    if labels is None:
        return [trim_name(n) for n in paramnames]

    if isinstance(labels, list):
        if len(labels) != len(paramnames):
            raise ValueError(
                f"labels list length ({len(labels)}) does not match "
                f"number of parameters ({len(paramnames)})"
            )
        return labels

    if isinstance(labels, dict):
        result = []
        for n in paramnames:
            short = n.split(".")[-1]
            if n in labels:
                result.append(labels[n])
            elif short in labels:
                result.append(labels[short])
            else:
                result.append(trim_name(n))
        return result

    raise TypeError(f"labels must be None, list, or dict, got {type(labels)}")



# ---------------------------------------------------------------------------
# CornerPlotter
# ---------------------------------------------------------------------------

class CornerPlotter:
    """
    Corner plot wrapper for UltraNest + 3ML posterior results.

    Parameters
    ----------
    ultranest_results : dict
        UltraNest result dict, e.g. loaded from results.json and patched
        with weighted_samples from weighted_post.txt.
    variates : dict
        3ML RandomVariates dict, e.g. from results.get_variates().
        Used to compute HPD and ETI intervals shown on the diagonal.
    """

    def __init__(self, ultranest_results: dict, variates: dict) -> None:
        self.ur = ultranest_results
        self.variates = variates

        self.hpds = [
            [
                par.highest_posterior_density_interval()[0],
                par.median,
                par.highest_posterior_density_interval()[1],
            ]
            for _, par in variates.items()
        ]
        self.etis = [
            [
                par.equal_tail_interval()[0],
                par.median,
                par.equal_tail_interval()[1],
            ]
            for _, par in variates.items()
        ]

        # also ETI = quantiles, just for double checking
        self.ur_meds   = np.array(self.ur["posterior"]["median"])
        self.ur_errlos = np.array(self.ur["posterior"]["errlo"])
        self.ur_errups = np.array(self.ur["posterior"]["errup"])



    def corner_plot(
        self,
        labels=None,
        fill: bool = True,
        datapoints: bool = False,
        log_inds: list = [],
        min_weight: float = 1e-4,
        with_legend: bool = True,
        levels: list = [0.6827, 0.9545, 0.9973],
        level_labels: list = [
            r"$68.3\%\,(1\sigma)$", r"$95.5\%\,(2\sigma)$", 
            r"$99.7\%\,(3\sigma)$"],
        contour_kwargs: dict = dict(
            linestyles=["-", "-.", ":"],
            colors=[_LINE_COLOR] * 3,
        ),
        color: str = _LINE_COLOR,
        quantiles: list = [0.15866, 0.5, 0.8413],
        eti_span: bool = False,
        eti_color: str = _ACCENT_COLOR,
        hpd_span: bool = False,
        hpd_color: str = _ACCENT_COLOR,
        ultranest_span: bool = False,
        ultranest_color: str = _ACCENT_COLOR,
        star: bool = True,
        lines_in_2D_histograms: bool = True,
        line_alpha = 0.5,
        **kwargs,
    ):
        
        """
        Produce a corner plot of the posterior.

        The diagonal shows 1D marginal histograms with optional credible
        interval overlays. The off-diagonal panels show 2D marginal contours.
        Three independent sources of 1σ interval estimates can be overlaid
        and compared: ETI from 3ML RandomVariates, HPD from 3ML RandomVariates,
        and the ETI quantiles stored directly by UltraNest in results.json.
        The title of each diagonal panel always shows the HPD interval.

        Parameters
        ----------
        labels : None | list | dict
            Parameter labels for the axes.
            - None  → auto-trimmed from paramnames via trim_name()
            - list  → used directly; must match number of parameters
            - dict  → maps short or full param name to label string;
                      unmapped names fall back to trim_name()
        fill : bool
            Fill contours in 2D panels. Default True.
        datapoints : bool
            Show individual posterior samples instead of filled contours.
            Overrides fill. Default False.
        log_inds : list of int
            Indices of parameters to display in log10 space. The data is
            transformed before plotting; tick labels are converted back to
            linear values. Default [].
        min_weight : float
            Cumulative weight threshold below which samples are masked.
            Removes negligible-weight points from the tails. Default 1e-4.
        with_legend : bool
            Add a legend above the top-right diagonal cell. Default True.
        levels : list of float
            Credible interval levels for the 2D contours, as enclosed
            probability fractions. Default [0.6827, 0.9545, 0.9973]
            (1σ, 2σ, 3σ).
        level_labels : list of str
            Legend labels corresponding to each entry in `levels`.
            Default [r"$68.3\\%\\,(1\\sigma)$", ...].
        contour_kwargs : dict
            Keyword arguments forwarded to the contour drawing in corner,
            e.g. linestyles and colors per level. Default: solid/dashdot/dotted
            lines in _LINE_COLOR.
        color : str
            Base colour for 1D histograms and 2D contour lines.
            Default _LINE_COLOR (#004488).
        quantiles : list of float
            Quantiles shown as vertical dashed lines on the 1D histograms.
            These are ETI bounds (equal-tail intervals = quantiles).
            Default [0.15866, 0.5, 0.8413] (16th, 50th, 84th percentile).
        eti_span : bool
            Overlay the 3ML ETI 1σ interval on the diagonal as a shaded span,
            and overplot the ETI median and bounds as lines in the 2D panels.
            Default False.
        eti_color : str
            Colour for ETI overlays. Default _ACCENT_COLOR (#BB5566).
        hpd_span : bool
            Overlay the 3ML HPD 1σ interval on the diagonal as a shaded span,
            and overplot the HPD median and bounds as lines in the 2D panels.
            For symmetric posteriors HPD == ETI; they differ for skewed ones.
            Default False.
        hpd_color : str
            Colour for HPD overlays. Default _ACCENT_COLOR (#BB5566).
        ultranest_span : bool
            Overlay the UltraNest ETI 1σ interval (errlo/errup from
            results.json, i.e. 15.87th and 84.13th percentiles) on the
            diagonal as a shaded span, and overplot the corresponding median
            and bounds as lines in the 2D panels. Default False.
        ultranest_color : str
            Colour for UltraNest ETI overlays. Default _ACCENT_COLOR (#BB5566).
        star : bool
            Overplot a star marker at the median position in the 2D panels,
            for whichever interval overlays are active. Default True.
        lines_in_2D_histograms : bool
            Draw median and bound lines across the 2D off-diagonal panels for
            active interval overlays. Set to False to show only the diagonal
            spans and star markers. Default True.
        **kwargs
            Additional keyword arguments forwarded to corner.corner.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The corner plot figure.

        Notes
        -----
        The diagonal panel titles always display the HPD interval from 3ML:
            label = value_{-lo}^{+hi}

        The three interval sources (ETI, HPD, UltraNest) can be enabled
        simultaneously for comparison. Since UltraNest ETI and 3ML ETI are
        computed from the same weighted samples, they should agree closely;
        any discrepancy indicates numerical differences in the quantile
        estimation. HPD will differ from ETI for skewed posteriors.
        """
        paramnames = self.ur["paramnames"]
        resolved_labels = _resolve_labels(paramnames, labels)

        data = np.array(self.ur["weighted_samples"]["points"])
        weights = np.array(self.ur["weighted_samples"]["weights"])

        # log-transform requested parameters
        for i in log_inds:
            data[:, i] = np.log10(data[:, i])

        # mask low-weight points
        cumsumweights = np.cumsum(weights)
        mask = cumsumweights > min_weight

        if mask.sum() == 1:
            raise ValueError(
                "Posterior is concentrated in a single point — "
                "try running the sampler longer."
            )


        # --- colormap ---
        cmap = kwargs.pop("cmap", _make_cmap(_CMAP_COLOR))
        contourf_kwargs = {"colors": None, "extend": "both", "cmap": cmap}

        # --- build corner kwargs ---
        kwargs.setdefault("plot_density", False)
        kwargs.setdefault("plot_datapoints", False)
        kwargs["labels"] = resolved_labels
        kwargs["show_titles"] = False
        kwargs["fill_contours"] = fill
        kwargs["levels"] = levels
        kwargs["quantiles"] = quantiles
        kwargs["contour_kwargs"] = contour_kwargs
        if fill:
            kwargs["contourf_kwargs"] = contourf_kwargs

        # suppress corner's small-dataset warning
        _orig_warn = logging.warning
        logging.warning = lambda *a, **kw: None

        if datapoints:
            fig = corner(
                data[mask],
                weights=weights[mask],
                plot_density=False,
                plot_datapoints=True,
                plot_contours=False,
                **kwargs,
            )
        else:
            fig = corner(
                data[mask],
                weights=weights[mask],
                color=color,
                **kwargs,
            )

        logging.warning = _orig_warn

        ndim = len(resolved_labels)
        axes = np.array(fig.axes).reshape((ndim, ndim))

        # --- recolour quantile lines on diagonal to accent colour ---
        # corner draws these as Line2D via axvline, inheriting `color`;
        # we identify them by checking they span the full axes height (0→1)
        for i in range(ndim):
            ax = axes[i, i]
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if len(ydata) == 2 and ydata[0] == 0.0 and ydata[1] == 1.0:
                    line.set_color(_ACCENT_COLOR)

        # --- diagonal: titles + ETI shading ---
        for i in range(ndim):
            ax = axes[i, i]
            low, med, up = self.hpds[i]
            ax.set_title(
                resolved_labels[i]
                + r"$=%.2f_{-%.2f}^{+%.2f}$" % (med, med - low, up - med),
                fontsize=9,
            )
            if eti_span:
                eti_lo, _, eti_hi = self.etis[i]
                span_lo = np.log10(eti_lo) if i in log_inds else eti_lo
                span_hi = np.log10(eti_hi) if i in log_inds else eti_hi
                ax.axvspan(span_lo, span_hi, color=eti_color, alpha=0.15)
            if hpd_span:
                hpd_lo, _, hpd_hi = self.hpds[i]
                span_lo = np.log10(hpd_lo) if i in log_inds else hpd_lo
                span_hi = np.log10(hpd_hi) if i in log_inds else hpd_hi
                ax.axvspan(span_lo, span_hi, color=hpd_color, alpha=0.15)
            if ultranest_span:
                un_lo, un_hi = self.ur_errlos[i], self.ur_errups[i]
                span_lo = np.log10(un_lo) if i in log_inds else un_lo
                span_hi = np.log10(un_hi) if i in log_inds else un_hi
                ax.axvspan(span_lo, span_hi, color=ultranest_color, alpha=0.15)


            # restore linear tick labels for log-transformed axes
            if i in log_inds:
                axes[ndim - 1, i].set_xticklabels(
                    [f"{10**t:.2g}" for t in axes[ndim - 1, i].get_xticks()]
                )
                axes[i, 0].set_yticklabels(
                    [f"{10**t:.2g}" for t in axes[i, 0].get_yticks()]
                )

        # --- overplot median lines + star ---
        if eti_span:
            errlos, meds, errups = [], [], []
            for i in range(len(self.etis)):
                eti_lo, med_i, eti_hi = self.etis[i]
                span_lo = np.log10(eti_lo) if i in log_inds else eti_lo
                med_i = np.log10(med_i) if i in log_inds else med_i
                span_hi = np.log10(eti_hi) if i in log_inds else eti_hi
                errlos.append(span_lo)
                meds.append(med_i)
                errups.append(span_hi)
            if lines_in_2D_histograms:
                overplot_lines(fig, meds,   color=eti_color, alpha=line_alpha)
                overplot_lines(fig, errlos, color=eti_color, alpha=line_alpha, ls="--")
                overplot_lines(fig, errups, color=eti_color, alpha=line_alpha, ls="--")
            if star:
                overplot_points(fig, [meds], marker="*", color=eti_color, ms=5)
        if hpd_span:
            errlos, meds, errups = [], [], []
            for i in range(len(self.hpds)):
                hpd_lo, med_i, hpd_hi = self.hpds[i]
                span_lo = np.log10(hpd_lo) if i in log_inds else hpd_lo
                med_i = np.log10(med_i) if i in log_inds else med_i
                span_hi = np.log10(hpd_hi) if i in log_inds else hpd_hi
                errlos.append(span_lo)
                meds.append(med_i)
                errups.append(span_hi)
            if lines_in_2D_histograms:
                overplot_lines(fig, meds,   color=hpd_color, alpha=line_alpha)
                overplot_lines(fig, errlos, color=hpd_color, alpha=line_alpha, ls="--")
                overplot_lines(fig, errups, color=hpd_color, alpha=line_alpha, ls="--")
            if star:
                overplot_points(fig, [meds], marker="*", color=hpd_color, ms=5)
        if ultranest_span:
            errlos, meds, errups = [], [], []
            for i in range(len(self.ur_meds)):
                ur_lo, med_i, ur_hi = self.ur_errlos[i], self.ur_meds[i], self.ur_errups[i]
                span_lo = np.log10(ur_lo) if i in log_inds else ur_lo
                med_i = np.log10(med_i) if i in log_inds else med_i
                span_hi = np.log10(ur_hi) if i in log_inds else ur_hi
                errlos.append(span_lo)
                meds.append(med_i)
                errups.append(span_hi)
            if lines_in_2D_histograms:
                overplot_lines(fig, meds,   color=ultranest_color, alpha=line_alpha)
                overplot_lines(fig, errlos, color=ultranest_color, alpha=line_alpha, ls="--")
                overplot_lines(fig, errups, color=ultranest_color, alpha=line_alpha, ls="--")
            if star:
                overplot_points(fig, [meds], marker="*", color=ultranest_color, ms=5)


        # --- legend ---
        if with_legend and ndim > 1:
            legend_handles = []

            # quantile lines on histogram — always shown when quantiles are set
            if quantiles:
                legend_handles.append(
                    mlines.Line2D(
                        [], [], linestyle="--", color=_ACCENT_COLOR,
                        label=r"$1\sigma$ (quantiles)",
                    )
                )
            if eti_span:
                legend_handles += [
                    mlines.Line2D(
                        [], [], linestyle="--", color=eti_color,
                        label=r"1$\sigma$ (ETI)",
                    ),
                ]
            if hpd_span:
                legend_handles += [
                    mlines.Line2D(
                        [], [], linestyle="--", color=hpd_color,
                        label=r"1$\sigma$ (HPD)",
                    ),
                ]
            if ultranest_span:
                legend_handles += [
                    mlines.Line2D(
                        [], [], linestyle="--", color=ultranest_color,
                        label=r"1$\sigma$ (UltraNest)",
                    ),
                ]

            # contour level entries — always shown
            legend_handles += [
                mlines.Line2D(
                    [], [], linestyle=ls, color=lc,
                    label=lbl,
                )
                for ls, lc, lbl in zip(
                    contour_kwargs.get("linestyles", []),
                    contour_kwargs.get("colors", [color] * 100),
                    level_labels
                )
            ]
            axes[ndim - 1, ndim - 1].legend(
                # title="credible prob level",
                handles=legend_handles,
                loc="lower left",
                bbox_to_anchor=(0, 1.3),
                # frameon=False,
                fontsize=10,
            )

        return fig

