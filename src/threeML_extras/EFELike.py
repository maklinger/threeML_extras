"""
EFELike.py — XYLike subclass for EFE data with nuisance parameter.

Likelihood is evaluated in EFE space [erg/cm²/s]:
    expectation_i = E_erg_i * E_keV_i * sum_sources(E_keV_i)

Mathematically equivalent to fitting in F(E) or dN/dE (E² cancels in chi²
when there are no finite bin widths).
"""
from typing import Union
import numpy as np

try:
    from threeML.io.logging import setup_logger
    from threeML.plugins.XYLike import XYLike, _chi2_like
except ImportError as e:
    raise ImportError("ThreeML not installed") from e
try:
    from astromodels.core.parameter import Parameter
    from astromodels.functions.priors import Uniform_prior
except ImportError as e:
    raise ImportError("astromodels not installed") from e

keV2erg = 1.60218e-9
log = setup_logger(__name__)


class EFELike(XYLike):
    """
    Plugin for EFE = E²F(E) data with chi² likelihood and optional
    effective area correction nuisance parameter.

    Inherits XYLike's masking, simulation, serialisation, fit(), plot().
    Overrides only get_log_like to apply the EFE model transform.

    Parameters
    ----------
    name : str
        Plugin name. Nuisance parameter will be named cons_<name>.
    x_keV : array-like
        Photon energies in keV.
    y_efe : array-like
        EFE values in erg/cm²/s.
    y_unc : array-like
        1-sigma uncertainties in erg/cm²/s.
    """

    def __init__(self, name, x_keV, y_efe, y_unc):
        self._x_erg = np.asarray(x_keV) * keV2erg
        super().__init__(name, x=x_keV, y=y_efe, yerr=y_unc)

        # XYLike.__init__ sets self._nuisance_parameters = {}
        # so we can safely add to it here
        self._nuisance_parameter = Parameter(
            "cons_%s" % name, 1.0,
            min_value=0.8, max_value=1.2, delta=0.05, free=False,
            desc="Effective area correction for %s" % name,
        )
        self._nuisance_parameters[self._nuisance_parameter.name] = self._nuisance_parameter

        log.info(
            f"{self.__class__.__name__}('{name}'): {len(self._x)} points, "
            f"E=[{self._x.min():.3e}, {self._x.max():.3e}] keV, "
            f"EFE=[{self._y.min():.3e}, {self._y.max():.3e}] erg/cm²/s"
        )

    def get_log_like(self):
        expectation = (self._x_erg * self._x * self._get_total_expectation())[self._mask]
        c = self._nuisance_parameter.value
        return _chi2_like(c * self._y[self._mask], c * self._yerr[self._mask], expectation)

    def use_effective_area_correction(self, 
                                      min_value: Union[int, float] = 0.8,
                                      max_value: Union[int, float] = 1.2) -> None:
        """Free the nuisance parameter with a uniform prior over [min_value, max_value]."""
        log.info(f"{self._name}: effective area correction enabled [{min_value}, {max_value}]")
        self._nuisance_parameter.free = True
        self._nuisance_parameter.bounds = (min_value, max_value)
        self._nuisance_parameter.set_uninformative_prior(Uniform_prior)

    def fix_effective_area_correction(self,
                                      value: Union[int, float] = 1) -> None:
        """Fix the effective area correction to value (default: 1)."""
        log.info(f"{self._name}: effective area correction fixed to {value}")
        self._nuisance_parameter.value = value
        self._nuisance_parameter.fix = True

    def add2ax_SED_eV_ergscm2(self, ax, color="k", **kwargs):
        """Plot E [eV] vs EFE [erg/cm²/s] onto ax."""
        ax.errorbar(self._x * 1e3, self._y, yerr=self._yerr,
                    ls="", marker=".", c=color, **kwargs)
        return ax