"""
SherpaPileupLike
================
A 3ML PluginPrototype that wraps a single Sherpa DataPHA dataset with a
JDPileup response.  The spectral model is driven by astromodels (via 3ML)
while the pileup kernel and likelihood are evaluated by Sherpa's compiled
C extension — giving exact Davis (2001) pileup at full MCMC speed.

Requirements
------------
- sherpa (standalone, or via CIAO)
- threeML / astromodels

Usage
-----
    plugin = SherpaPileupLike(
        name        = "core",
        pha_file    = "core_21075.pha",
        arf_file    = "core_21075.arf",
        rmf_file    = "core_21075.rmf",
        bkg_file    = "core_21075_bkg.pha",   # optional
        alpha       = 0.5,      # grade migration — free parameter
        f           = 0.95,     # PSF fraction    — usually frozen
        n           = 1,        # number of regions — usually frozen
        stat        = "wstat",  # Sherpa statistic
    )
"""

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
    from sherpa.astro.instrument import PileupResponse1D
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
    pileup kernel and the C-stat likelihood are evaluated by Sherpa's compiled
    C extension (``sherpa.astro.utils.apply_pileup``).

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
        Grade-migration parameter (0–1).  This is the only free pileup
        parameter in most analyses.
    g0 : float
        Single-photon grade-zero probability.  Frozen by default.
    f : float
        Fraction of PSF flux in the pileup region.  Frozen by default.
    n : int
        Number of detection regions.  Always frozen.
    nterms : int
        Maximum number of piled photons to consider.  Always frozen.
    stat : str
        Sherpa fit statistic.  ``"wstat"`` (default) or ``"cstat"``.
    """

    def __init__(
        self,
        name: str,
        pha_file: str,
        arf_file: str,
        rmf_file: str,
        bkg_file: Optional[str] = None,
        ftime: float = 3.241,
        fracexp: float = 0.987,
        alpha: float = 0.5,
        g0: float = 1.0,
        f: float = 0.95,
        n: int = 1,
        nterms: int = 30,
        e_lo_keV: float = 0.5,
        e_hi_keV: float = 7.0,
        stat: str = "wstat",
    ):
        if not _SHERPA_AVAILABLE:
            raise ImportError(
                "Sherpa is required for SherpaPileupLike. "
                "Install it with: conda install -c cxc sherpa"
            )

        # ── Assign a unique Sherpa dataset ID ────────────────────────────────
        self._sherpa_id = _next_dataset_id()
        sid = self._sherpa_id

        # ── Load data into Sherpa ─────────────────────────────────────────────
        shp.load_pha(sid, pha_file)
        shp.load_arf(sid, arf_file)
        shp.load_rmf(sid, rmf_file)
        if bkg_file is not None:
            shp.load_bkg(sid, bkg_file)

        # Read from headers
        pha_hdr = shp.get_data(sid).header
        ftime = pha_hdr.get("TIMEDEL", pha_hdr.get("EXPTIME"))
        if ftime is None:
            raise ValueError(f"Cannot read frame time from {pha_file}. Supply ftime manually.")

        arf_hdr = shp.get_arf(sid).header
        fracexp = arf_hdr.get("FRACEXPO")
        if fracexp is None:
            raise ValueError(f"Cannot read FRACEXPO from {arf_file}. Supply fracexp manually.")

        # ── Energy filter ─────────────────────────────────────────────────────
        shp.notice_id(sid, e_lo_keV, e_hi_keV)

        # ── Sherpa statistic ──────────────────────────────────────────────────
        self._stat = stat
        shp.set_stat(stat)

        # ── Build the JDPileup model ──────────────────────────────────────────
        self._jdp = JDPileup(f"jdp_{name}")
        self._jdp.alpha.val   = alpha
        self._jdp.g0.val      = g0
        self._jdp.f.val       = f
        self._jdp.n.val       = n
        self._jdp.ftime.val   = ftime
        self._jdp.fracexp.val = fracexp
        self._jdp.nterms.val  = nterms

        # Freeze everything except alpha by default
        self._jdp.alpha.frozen   = False
        self._jdp.g0.frozen      = True
        self._jdp.f.frozen       = True
        self._jdp.n.frozen       = True
        self._jdp.ftime.frozen   = True
        self._jdp.fracexp.frozen = True
        self._jdp.nterms.frozen  = True

        # ── TableModel: astromodels drives the flux, Sherpa sees a table ──────
        # The table is updated at every likelihood call via _sync_model().
        arf      = shp.get_arf(sid)
        self._e_lo   = np.array(arf.energ_lo)
        self._e_hi   = np.array(arf.energ_hi)
        self._n_bins = len(self._e_lo)

        self._table = TableModel(f"src_{name}")
        self._table.load(self._e_lo, np.ones(self._n_bins))
        self._table.ampl = 1.0
        self._table.ampl.frozen = True

        # ── Wrap with PileupResponse1D (ARF + pileup + RMF) ──────────────────
        pha_data = shp.get_data(sid)
        rsp      = PileupResponse1D(pha_data, self._jdp)
        shp.set_full_model(sid, rsp(self._table))

        # ── Expose alpha as a 3ML nuisance parameter ──────────────────────────
        # astromodels Parameter wrapping the Sherpa alpha
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
        # Evaluate differential flux at bin midpoints, multiply by bin width
        e_mid = 0.5 * (self._e_lo + self._e_hi)
        de    = self._e_hi - self._e_lo
        flux = np.zeros(self._n_bins)
        for i in range(self._like_model.get_number_of_point_sources()):
            flux += self._like_model.get_point_source_fluxes(i, e_mid) * de
        self._table.load(self._e_lo, flux)
        self._jdp.alpha.val = self._alpha_par.value

    # ── PluginPrototype interface ─────────────────────────────────────────────

    def set_model(self, likelihood_model_instance: Model) -> None:
        self._like_model = likelihood_model_instance


    def get_log_like(self) -> float:
        """Evaluate the Sherpa C-stat log-likelihood with current parameters."""
        shp.set_stat(self._stat) # to avoid clashes with other statistics/sessions
        self._sync_model()
        # Sherpa calc_stat returns the *statistic value* (lower = better),
        # so we negate it to get a log-likelihood (higher = better).
        return -shp.calc_stat(self._sherpa_id)

    def inner_fit(self) -> float:
        """Profile likelihood: no internal minimisation needed here."""
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

    def set_active_measurements(self, e_lo_keV: float, e_hi_keV: float) -> None:
        """Change the active energy range after construction."""
        shp.notice_id(self._sherpa_id, e_lo_keV, e_hi_keV)

    def freeze_alpha(self, value: Optional[float] = None) -> None:
        """Freeze the alpha parameter, optionally setting a new value."""
        if value is not None:
            self._alpha_par.value = value
        self._alpha_par.free = False

    def free_alpha(self) -> None:
        """Free the alpha parameter for fitting."""
        self._alpha_par.free = True

    def display_pileup_fractions(self) -> None:
        """Print the per-term pileup fractions (calls Sherpa __str__)."""
        # Trigger one evaluation to populate _results
        self._sync_model()
        shp.calc_stat(self._sherpa_id)
        print(str(self._jdp))
