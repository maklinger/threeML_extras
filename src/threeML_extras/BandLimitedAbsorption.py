"""
Factory for creating band-limited wrappers around astromodels absorption models.

Absorption models are typically only defined over a finite energy range. Outside
that range they may return unphysical values (e.g. very small but non-zero
transmission). This module provides a factory function that wraps any
astromodels Function1D absorption model and clamps its output to 1.0 (i.e. no
absorption) outside a user-defined energy range [e_min, e_max].

Usage
-----
    from BandLimitedAbsorption import make_band_limited
    from astromodels import ZDust

    BandLimitedZDust = make_band_limited(ZDust, e_min=1e-6, e_max=1.5e-2)
    f = BandLimitedZDust(e_bmv=1.0, rv=3.1)
    f(5e-4)  # evaluated normally, within range
    f(1.0)   # returns 1.0, outside range

The returned class is a proper astromodels Function1D subclass and is fully
compatible with 3ML: it can be used in spectral models, fitted, and
saved/loaded. All parameters of the original model are preserved.

The energy bounds e_min and e_max are plain instance attributes (not
astromodels parameters) and are set at class creation time. They can be
updated on an instance at any time:

    f._e_min = 1e-5  # keV
    f._e_max = 2e-2  # keV
"""

import numpy as np
from astromodels import Function1D


def make_band_limited(absorption_cls, e_min=0.1, e_max=100.0):
    """
    Create a band-limited version of an astromodels absorption model.

    The returned class behaves identically to ``absorption_cls`` within
    [e_min, e_max], and returns 1.0 (no absorption) outside that range.

    Parameters
    ----------
    absorption_cls : type
        Any astromodels Function1D absorption class (e.g. ZDust, TbAbs).
    e_min : float, optional
        Lower energy bound in keV. Absorption is set to 1.0 below this value.
        Default is 0.1 keV.
    e_max : float, optional
        Upper energy bound in keV. Absorption is set to 1.0 above this value.
        Default is 100.0 keV.

    Returns
    -------
    type
        A new astromodels Function1D subclass named
        ``BandLimited{absorption_cls.__name__}`` with the same parameters
        as the original model. The bounds e_min and e_max are stored as
        ``_e_min`` and ``_e_max`` (in keV) on each instance and can be
        updated after construction.

    Examples
    --------
    >>> from astromodels import ZDust
    >>> BandLimitedZDust = make_band_limited(ZDust, e_min=1e-6, e_max=1.5e-2)  # keV
    >>> f = BandLimitedZDust(e_bmv=1.0, rv=3.1)
    >>> f(5e-4)   # within range: normal ZDust evaluation
    >>> f(1.0)    # outside range: returns 1.0
    """

    # Instantiate once to introspect parameter names
    _tmp = absorption_cls()
    param_names = list(_tmp._parameters.keys())

    # Build evaluate() dynamically with the exact parameter signature that
    # FunctionMeta requires — *args does not satisfy its validation
    sig = ", ".join(["x"] + param_names)
    src = f"""
def evaluate(self, {sig}):
    result = absorption_cls.evaluate(self, {sig})
    return np.where((x >= self._e_min) & (x <= self._e_max), result, 1.0)
"""
    globs = {"np": np, "absorption_cls": absorption_cls}
    exec(src, globs)

    def _setup(self):
        super(BandLimited, self)._setup()
        self._e_min = e_min  # keV
        self._e_max = e_max  # keV

    def _set_units(self, x_unit, y_unit):
        super(BandLimited, self)._set_units(x_unit, y_unit)

    BandLimited = type(
        f"BandLimited{absorption_cls.__name__}",
        (absorption_cls,),
        {
            "__doc__": absorption_cls.__doc__,
            "_setup": _setup,
            "_set_units": _set_units,
            "evaluate": globs["evaluate"],
        },
    )

    return BandLimited