import collections
from typing import Optional

import numpy as np
from astromodels import Model
from astromodels.core.parameter import Parameter

from threeML.plugin_prototype import PluginPrototype

# Sherpa imports — all lazy so the class can be imported without Sherpa
try:
    import sherpa.astro.ui as shp
    from sherpa.astro.data import DataPHA
    from sherpa.astro.models import JDPileup
    from sherpa.models import TableModel

    _SHERPA_AVAILABLE = True
except ImportError:
    _SHERPA_AVAILABLE = False


# ── Dataset ID counter so multiple instances don't collide ───────────────────
_DATASET_COUNTER = 0


def _next_dataset_id() -> int:
    global _DATASET_COUNTER
    _DATASET_COUNTER += 1
    return _DATASET_COUNTER


class SherpaPileupLike(PluginPrototype):
    """3ML plugin for a piled-up Chandra ACIS spectrum.

    The spectral model is provided by astromodels / 3ML.  The Davis (2001)
    pileup kernel and the likelihood are evaluated by Sherpa's compiled
    C extension (``sherpa.astro.utils.apply_pileup``).

    The response chain is built via Sherpa's native API:
        set_source(sid, table)
        set_pileup_model(sid, jdp)
    which produces:  apply_rmf(jdp(apply_arf(table)))

    Parameters
    ----------
    name : str
        Plugin name (must be a valid Python identifier).
    pha_file : str
        Path to the source PHA file.
    arf_file : str
        Path to the ARF file.
    rmf_file : str
        Path to the RMF file.
    bkg_file : str, optional
        Path to the background PHA file.
    alpha : float
        Grade-migration parameter (0-1).
    g0 : float
        Single-photon grade-zero probability.  Frozen by default.
    f : float
        Fraction of PSF flux in the pileup region.  Frozen by default.
    n : int
        Number of detection regions.  Always frozen.
    nterms : int
        Maximum number of piled photons to consider.  Always frozen.
    ftime : float
        The frame time in seconds - can be read from the header too.
    e_lo_keV : float
        Lower energy bound for noticed channels [keV].
    e_hi_keV : float
        Upper energy bound for noticed channels [keV].
    stat : str
        Sherpa fit statistic.  ``"wstat"`` (default) or ``"cstat"``.
    pileup : bool
        Set the pileup model or not.
    """

    def __init__(
        self,
        name: str,
        pha_file: str,
        arf_file: str,
        rmf_file: str,
        bkg_file: Optional[str] = None,
        alpha: float = 0.5,
        g0: float = 1.0,
        f: float = 0.95,
        n: int = 1,
        nterms: int = 30,
        ftime: float = 0.4,
        e_lo_keV: float = 0.5,
        e_hi_keV: float = 7.0,
        stat: str = "wstat",
        pileup: bool = True
    ):
        if not _SHERPA_AVAILABLE:
            raise ImportError(
                "Sherpa is required for SherpaPileupLike. "
                "Install it with: conda install -c cxc sherpa"
            )

        # ── Assign a unique Sherpa dataset ID ────────────────────────────────
        self._sherpa_id = _next_dataset_id()
        sid = self._sherpa_id

        # ── Store energy limits for re-application after regrouping ──────────
        self._e_lo_keV = e_lo_keV
        self._e_hi_keV = e_hi_keV

        # ── Load data into Sherpa ─────────────────────────────────────────────
        shp.load_pha(sid, pha_file)
        shp.load_arf(sid, arf_file)
        shp.load_rmf(sid, rmf_file)
        if bkg_file is not None:
            shp.load_bkg(sid, bkg_file)

        # ── Read frame time from PHA header ───────────────────────────────────
        # EXPTIME is the static per-frame exposure time (e.g. 3.2 s for
        # full-frame TIMED mode), as defined in the CIAO times documentation.
        # TIMEDEL = EXPTIME + 0.04104 s (includes charge transfer) and is NOT
        # the correct quantity for JDPileup.ftime.

        pha_hdr = shp.get_data(sid).header
        if ftime is None:
            ftime = pha_hdr.get("EXPTIME")
            _ftime_source = "EXPTIME"

            if ftime is None:
                ftime = pha_hdr.get("TIMEDEL") - 0.04104
                _ftime_source = "TIMEDEL (includes 0.04104 s charge transfer — slightly overestimates ftime). Subtracted 0.04104 s."

            if ftime is None:
                ftime = 3.2
                _ftime_source = "DEFAULT 3.2 s (full-frame ACIS TIMED mode — verify this is correct for your data!)"
                print(
                    f"[{name}] WARNING: EXPTIME and TIMEDEL not found in PHA header. "
                    f"Defaulting to ftime={ftime} s. "
                    f"Available header keys: {list(pha_hdr.keys()) if hasattr(pha_hdr, 'keys') else pha_hdr}"
                )
        else:
            _ftime_source = ftime

        print(f"[{name}] ftime={ftime:.5f} s  (source: {_ftime_source})")

        # ── Read FRACEXPO from ARF header ─────────────────────────────────────
        arf_hdr = shp.get_arf(sid).header
        fracexp = arf_hdr.get("FRACEXPO")
        if fracexp is None:
            print(f"[{name}] Cannot read FRACEXPO from {arf_file}. Defaulting to 1.0.")
            fracexp = 1.0

        print(f"[{name}] ftime={ftime:.5f} s  (EXPTIME), fracexp={fracexp:.4f}")

        # ── Build the JDPileup model ──────────────────────────────────────────
        self._jdp = JDPileup(f"jdp_{name}")
        self._jdp.alpha.val    = alpha
        self._jdp.g0.val       = g0
        self._jdp.f.val        = f
        self._jdp.n.val        = n
        self._jdp.ftime.val    = ftime
        self._jdp.fracexp.val  = fracexp
        self._jdp.nterms.val   = nterms

        # Freeze everything except alpha by default
        self._jdp.alpha.frozen   = True
        self._jdp.g0.frozen      = True
        self._jdp.f.frozen       = True
        self._jdp.n.frozen       = True
        self._jdp.ftime.frozen   = True
        self._jdp.fracexp.frozen = True
        self._jdp.nterms.frozen  = True

        # ── TableModel: astromodels drives the flux, Sherpa sees a table ──────
        # Initialised on the ARF energy grid (bin edges); updated at every
        # likelihood call. 
        arf = shp.get_arf(sid)
        self._e_lo   = np.array(arf.energ_lo)
        self._e_hi   = np.array(arf.energ_hi)
        self._n_bins = len(self._e_lo)

        self._table = TableModel(f"src_{name}")
        self._table.load(self._e_lo, np.ones(self._n_bins)) # some init value
        self._table.ampl        = 1.0
        self._table.ampl.frozen = True

        shp.set_source(sid, self._table)
        if pileup:
            shp.set_pileup_model(sid, self._jdp)

        shp.set_stat(stat)

        # ── Store stat name ───────────────────────────────────────────────────
        self._stat = stat

        # ── Expose alpha as a 3ML nuisance parameter ──────────────────────────
        self._alpha_par = Parameter(
            f"alpha_{name}",
            alpha,
            min_value=0.0,
            max_value=1.0,
            delta=0.05,
            free=True,
            desc="JDPileup grade-migration parameter",
        )

        nuisance = collections.OrderedDict()
        nuisance[self._alpha_par.name] = self._alpha_par

        # ── Store reference to the 3ML likelihood model (set later) ──────────
        self._like_model: Optional[Model] = None

        super().__init__(name, nuisance)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _sync_model(self) -> None:
        """Push current astromodels flux into the Sherpa TableModel.

        astromodels' __call__(e_mid) returns differential photon flux
        [photons/cm²/s/keV] at bin midpoints. We multiply by bin widths
        to get integrated counts-equivalent flux per bin, then load onto
        the ARF energy grid for Sherpa's TableModel.
        """
        if self._like_model is None:
            raise RuntimeError("set_model() has not been called yet.")

        if self._like_model.get_number_of_extended_sources() > 0:
            raise NotImplementedError(
                "SherpaPileupLike does not support extended sources."
            )

        e_mid = 0.5 * (self._e_lo + self._e_hi)
        d_e   = self._e_hi - self._e_lo

        flux = np.zeros(self._n_bins)
        for i in range(self._like_model.get_number_of_point_sources()):
            # get_point_source_fluxes with a single array returns
            # differential flux [photons/cm²/s/keV] at each midpoint
            flux += self._like_model.get_point_source_fluxes(i, e_mid) * d_e

        self._table.load(e_mid, flux)
        self._jdp.alpha.val = self._alpha_par.value


    # from the 3ML experimental SherpaLike
    # def _sync_model(self) -> None:
    #     vals = np.zeros(self._n_bins)
    #     for i in range(self._like_model.get_number_of_point_sources()):
    #         # get_point_source_fluxes returns differential flux [ph/cm²/s/keV]
    #         # multiply by bin width to get integrated flux [ph/cm²/s] per bin
    #         e_mid = 0.5 * (self._e_lo + self._e_hi)
    #         d_e = self._e_hi - self._e_lo
    #         vals += self._like_model.get_point_source_fluxes(i, e_mid) * d_e

    #     self._table._TableModel__x = self._e_lo
    #     self._table._TableModel__y = vals
    #     self._table.cache_clear()
    #     self._jdp.alpha.val = self._alpha_par.value

    # ── PluginPrototype interface ─────────────────────────────────────────────

    def set_model(self, likelihood_model_instance: Model) -> None:
        self._like_model = likelihood_model_instance

    def get_log_like(self) -> float:
        """Evaluate the Sherpa log-likelihood with current parameters.

        Sherpa's calc_stat returns the Cash / W-stat value, which equals
        -2 * ln(L).  We therefore divide by -2 to obtain ln(L).
        """
        shp.set_stat(self._stat)  # guard against cross-session contamination
        self._sync_model()
        return -0.5 * shp.calc_stat(self._sherpa_id)

    def inner_fit(self) -> float:
        """Profile likelihood: no internal minimisation needed."""
        return self.get_log_like()

    # ── Convenience ──────────────────────────────────────────────────────────

    def get_number_of_data_points(self) -> int:
        return int(shp.get_data(self._sherpa_id).get_dep(filter=True).size)

    @property
    def alpha(self) -> Parameter:
        """The grade-migration nuisance parameter (astromodels Parameter)."""
        return self._alpha_par

    @property
    def jdpileup(self) -> "JDPileup":
        """Direct access to the underlying Sherpa JDPileup model."""
        return self._jdp

    def set_active_measurements(self, energy_range: str) -> None:
        """Set the active energy range, e.g. '0.4-8.0' [keV]."""
        lo, hi = [float(x) for x in energy_range.split("-")]
        self._e_lo_keV = lo
        self._e_hi_keV = hi
        shp.ignore_id(self._sherpa_id)
        shp.notice_id(self._sherpa_id, lo, hi)
        print(f"updated energy range to {lo} - {hi}keV.")

    def freeze_alpha(self, value: Optional[float] = None) -> None:
        """Freeze alpha, optionally setting a new value."""
        if value is not None:
            self._alpha_par.value = value
        self._alpha_par.free = False

    def free_alpha(self) -> None:
        """Free alpha for fitting."""
        self._alpha_par.free = True

    def display_pileup_fractions(self) -> None:
        """Print per-term pileup fractions (populates JDPileup._results)."""
        self._sync_model()
        shp.calc_stat(self._sherpa_id)
        print(str(self._jdp))

    def get_exposure(self) -> float:
        return shp.get_data(self._sherpa_id).exposure

    def get_energy_channels(self):
        """Return (e_lo, e_mid, e_hi) arrays for the current grouped bins."""
        dp    = shp.get_data_plot(self._sherpa_id)
        return dp.xlo, dp.x, dp.xhi
    
    def get_data_rate(self):
        """ Diff. rate and uncertainty in 1/(s*keV)"""
        dp = shp.get_data_plot(self._sherpa_id)
        return dp.y, dp.yerr
    
    def get_model_rate(self):
        """ Diff. rate and uncertainty in 1/(s*keV)"""
        self._sync_model()
        print("updated")
        shp.calc_stat(self._sherpa_id)  # force pileup kernel evaluation
        fp = shp.get_fit_plot(self._sherpa_id)
        return fp.modelplot.y


    def rebin(self, threshold: int = 3, strategy: str = "snr") -> None:
        """Rebin the spectrum and re-apply the energy filter.

        Parameters
        ----------
        threshold : int or float
            Binning threshold (meaning depends on strategy).
        strategy : str
            One of ``"snr"``, ``"min_counts"``, or ``"bins"``.
        """
        if strategy.lower() == "snr":
            shp.group_snr(self._sherpa_id, threshold)
        elif strategy.lower() == "min_counts":
            shp.group_counts(self._sherpa_id, threshold)
        elif strategy.lower() == "bins":
            shp.group_bins(self._sherpa_id, threshold)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from 'snr', 'min_counts', or 'bins'."
            )
        # Grouping resets the noticed range — restore it.
        shp.notice_id(self._sherpa_id, self._e_lo_keV, self._e_hi_keV)

    def ungroup(self) -> None:
        """Remove grouping and restore the energy filter."""
        shp.ungroup(self._sherpa_id)
        shp.notice_id(self._sherpa_id, self._e_lo_keV, self._e_hi_keV)
