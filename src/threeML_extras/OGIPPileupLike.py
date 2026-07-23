import numpy as np
from numba import njit
from astromodels.core.parameter import Parameter
from threeML.plugins.OGIPLike import OGIPLike

from threeML.io.logging import setup_logger
log = setup_logger(__name__)

class OGIPPileupLike(OGIPLike):

    def __init__(
        self,
        name,
        observation,
        background=None,
        response=None,
        arf_file=None,
        alpha=0.5,
        f=0.95,
        n_regions=1,
        g0=1.0,
        nterms=30,
        ftime=None,
        fracexpo=None,
        verbose=True,
    ):
        super().__init__(
            name=name,
            observation=observation,
            background=background,
            response=response,
            arf_file=arf_file,
            verbose=verbose,
        )

        rsp = self._response
        if rsp.arf is None:
            raise RuntimeError(f"[{name}] No ARF loaded. OGIPLikePileup requires a separate ARF file.")

        mc = rsp.monte_carlo_energies
        self._arf_energ_lo = mc[:-1]
        self._arf_energ_hi = mc[1:]
        self._specresp     = rsp.arf.astype(float)
        self._rmf          = rsp.rmf.astype(float)

        self._ftime    = float(ftime)    if ftime    is not None else self._read_ftime_from_header()
        self._fracexpo = float(fracexpo) if fracexpo is not None else self._read_fracexpo_from_header()
        self._g0        = float(g0)
        self._n_regions = int(n_regions)
        self._nterms    = int(nterms)

        self._alpha_par = Parameter(
            f"alpha_{name}", alpha,
            min_value=0.0, max_value=1.0, delta=0.05, free=True,
            desc=f"JDPileup grade-migration parameter for {name}",
        )
        self._f_par = Parameter(
            f"f_{name}", f,
            min_value=0.0, max_value=1.0, delta=0.01, free=False,
            desc=f"JDPileup PSF fraction for {name}",
        )

        self._nuisance_parameters[self._alpha_par.name] = self._alpha_par
        self._nuisance_parameters[self._f_par.name]     = self._f_par

        self._last_pileup_fractions = None
        self._last_integral_ae      = None
        self._last_num_terms        = None
        self._differential_flux     = None

        if verbose:
            log.info(f"[{name}] OGIPLikePileup ready: ftime={self._ftime:.4f} s, "
                    f"fracexpo={self._fracexpo:.4f}, alpha={alpha} (free), f={f} (frozen)")

            
    def _read_ftime_from_header(self):
        try:
            import astropy.io.fits as fits
            with fits.open(self._observed_spectrum._file_name) as f:
                hdr = f["SPECTRUM"].header
                
                val = hdr.get("EXPTIME", None)
                if val is not None:
                    log.info(f"[{self._name}] ftime={float(val):.5f} s (source: EXPTIME)")
                    return float(val)
                
                val = hdr.get("TIMEDEL", None)
                if val is not None:
                    corrected = float(val) - 0.04104
                    log.warning(
                        f"[{self._name}] EXPTIME not found, using TIMEDEL - 0.04104 s "
                        f"(subtracting charge transfer time): {corrected:.5f} s"
                    )
                    return corrected

        except Exception as e:
            log.warning(f"[{self._name}] Could not read PHA header ({e}), defaulting to 3.2 s")

        log.warning(f"[{self._name}] EXPTIME/TIMEDEL not found, defaulting to 3.2 s")
        return 3.2

    def _read_fracexpo_from_header(self):
        try:
            import astropy.io.fits as fits
            arf_file = self._response._arf_file   # ← correct attribute
            with fits.open(str(arf_file)) as f:
                val = f["SPECRESP"].header.get("FRACEXPO", None)
            if val is not None:
                log.info(f"[{self._name}] fracexpo={float(val):.5f} (source: FRACEXPO from ARF)")
                return float(val)
        except Exception as e:
            log.warning(f"[{self._name}] Could not read ARF header ({e}), defaulting fracexpo=1.0")
        return 1.0


    def set_model(self, likelihood_model):
        super().set_model(likelihood_model)
        self._differential_flux, _ = self._get_diff_flux_and_integral(
            self._like_model,
            integrate_method=self._model_integrate_method,
        )

    def _evaluate_model(self, precalc_fluxes=None):
        if self._differential_flux is None:
            raise RuntimeError("set_model() has not been called yet.")

        model_flux = eval_model_on_internal_grid(self._differential_flux)

        piled_counts, pileup_fractions, integral_ae, num_terms = apply_pileup_numpy(
            model_flux    = model_flux,
            energ_lo      = self._arf_energ_lo,
            energ_hi      = self._arf_energ_hi,
            specresp      = self._specresp,
            exposure_time = self.exposure,
            frame_time    = self._ftime,
            fracexpo      = self._fracexpo,
            alpha         = self._alpha_par.value,
            g0            = self._g0,
            num_regions   = self._n_regions,
            psf_frac      = self._f_par.value,
            max_num_terms = self._nterms,
        )

        self._last_pileup_fractions = pileup_fractions
        self._last_integral_ae      = integral_ae
        self._last_num_terms        = num_terms

        counts_per_channel = np.dot(self._rmf, piled_counts)
        return counts_per_channel / self.exposure

    @property
    def alpha(self):
        return self._alpha_par

    @property
    def f(self):
        return self._f_par

    def freeze_alpha(self, value=None):
        if value is not None:
            self._alpha_par.value = value
        self._alpha_par.free = False

    def free_alpha(self):
        self._alpha_par.free = True

    def freeze_f(self, value=None):
        if value is not None:
            self._f_par.value = value
        self._f_par.free = False

    def free_f(self):
        self._f_par.free = True

    def display_pileup_fractions(self):
        if self._last_pileup_fractions is None:
            log.warning("No pileup evaluation yet — call get_log_like() first.")
            return
        pf, ae, nt = self._last_pileup_fractions, self._last_integral_ae, self._last_num_terms
        total = pf[1:nt + 1].sum()
        if total == 0:
            log.warning("Pileup fraction: 0")
            return
        print(f"integral_ae (μ/g0) = {ae:.4f}")
        print(f"{'term':>5}  {'Poisson prob':>14}  {'fraction':>10}")
        pn = np.exp(-ae)
        for i in range(1, nt + 1):
            pn *= ae / i
            print(f"{i:>5}  {pn:>14.4e}  {pf[i]/total:>10.4f}")
        print(f"\n*** pileup fraction: {pf[2:nt+1].sum()/total:.4f}")

    @property
    def pileup_fraction(self):
        if self._last_pileup_fractions is None:
            return None
        pf, nt = self._last_pileup_fractions, self._last_num_terms
        total = pf[1:nt + 1].sum()
        return float(pf[2:nt + 1].sum() / total) if total > 0 else 0.0

    def _output(self):
        import pandas as pd
        base = super()._output()
        info = {
            "pileup alpha":     self._alpha_par.value,
            "pileup f":         self._f_par.value,
            "pileup ftime":     self._ftime,
            "pileup fracexpo":  self._fracexpo,
            "pileup g0":        self._g0,
            "pileup n_regions": self._n_regions,
        }
        if self._last_integral_ae is not None:
            info["pileup integral_ae"] = self._last_integral_ae
            info["pileup fraction"]    = self.pileup_fraction
        return pd.concat([base, pd.Series(info)])











"""
Pure NumPy translation of Sherpa's pileup.cc (John E. Davis, MIT, 2000).

Key design choices vs. the C original:
- model is pre-evaluated outside and passed as an array (avoids Numba callback issue)
- FFT done with np.fft.rfft (equivalent to JDMfftn for real input)
- rebin_histogram translated directly
- internal grid is identical: 0–15 keV, NUM_POINTS=4096

Units (matching the C code exactly):
    specresp      : cm²           (ARF effective area, per MC bin)
    frame_time    : s             (EXPTIME keyword)
    exposure_time : s             (total observation exposure)
    fracexpo      : dimensionless (FRACEXPO from ARF header)
    model_flux    : ph/cm²/s/keV  (differential photon flux, evaluated on
                                   the internal 4096-point grid enlo/enhi)
    returns       : ph/frame      (piled-up counts per detector bin,
                                   before RMF fold)
"""


NUM_POINTS = 1024 * 4          # 4096 — hard-coded in C
E_MIN      = 0.0
E_MAX      = 15.0              # keV — hard-coded in C
MAX_PROBABILITY_CUTOFF = 0.99999


# ── Internal grid ─────────────────────────────────────────────────────────────

def _make_internal_grid():
    """Build the fixed 4096-point energy grid (0–15 keV)."""
    de = (E_MAX - E_MIN) / NUM_POINTS          # ~0.003662 keV
    energies = np.arange(NUM_POINTS) * de
    energies[0] = 1e-4                         # must be non-zero (C code)

    # bin edges: enlo[i] = energies[i-1], enhi[i] = energies[i]
    # (matches the C convert_spectrum logic)
    enhi = energies.copy()
    enlo = np.empty(NUM_POINTS)
    enlo[0] = energies[0] / 2.0 if de >= energies[0] else energies[0]
    enlo[1:] = energies[:-1]
    enhi[-1] = energies[-1] + de

    return energies, enlo, enhi, de


_ENERGIES, _ENLO, _ENHI, _DE = _make_internal_grid()
_E_MID  = 0.5 * (_ENLO + _ENHI)
_E_EDGES = np.append(_ENLO, _ENHI[-1])   # shape (4097,) — edges only


# ── ARF resampling ────────────────────────────────────────────────────────────

def _build_arf_time(energ_lo, energ_hi, specresp, frame_time, fracexpo):
    """
    Resample specresp onto the internal 4096-point grid and multiply by
    frame_time.  Matches init_kernel() in C.

    Returns arf_time [cm²·s], shape (NUM_POINTS,)
    """
    min_e = energ_lo[0]
    max_e = energ_hi[-1]

    arf_time = np.zeros(NUM_POINTS)
    for i in range(1, NUM_POINTS):
        e = _ENERGIES[i]
        if e < min_e or e >= max_e:
            continue
        idx = np.searchsorted(energ_lo, e, side='right') - 1
        idx = np.clip(idx, 0, len(specresp) - 1)
        val = specresp[idx] * frame_time
        arf_time[i] = max(val, 0.0)

    return arf_time


# ── Histogram rebinning ───────────────────────────────────────────────────────
@njit(cache=True)
def _rebin_histogram(fy, flo, fhi, tlo, thi):
    """
    Direct translation of rebin_histogram() from C.
    Integrates fy(flo,fhi) onto target bins (tlo,thi) by overlap fraction.

    Returns ty, shape (len(tlo),)
    """
    nf = len(flo)
    nt = len(tlo)
    ty = np.zeros(nt)
    f = 0
    for t in range(nt):
        t0, t1 = tlo[t], thi[t]
        s = 0.0
        ff = f
        while ff < nf:
            f0, f1 = flo[ff], fhi[ff]
            if t0 > f1:
                ff += 1
                continue
            if f0 > t1:
                break
            max_min = max(t0, f0)
            min_max = min(t1, f1)
            if f0 == f1:
                return None   # division by zero guard (matches C)
            s += fy[ff] * (min_max - max_min) / (f1 - f0)
            if f1 > t1:
                break
            ff += 1
        ty[t] = s
    return ty


# ── FFT convolution ───────────────────────────────────────────────────────────

def _fft_convolve(fft_s, s):
    """
    Convolve s with the already-FFT'd template fft_s.
    Matches do_convolution() in C (zero-padded, linear convolution).

    fft_s : rfft of the normalised arf_s_tmp (length NUM_POINTS)
    s     : current arf_s_tmp (modified in-place, returns left half)
    """
    n = len(s)
    # zero-pad to 2n, take rfft
    s_padded = np.zeros(2 * n)
    s_padded[n:] = s
    S = np.fft.rfft(s_padded)
    # multiply in frequency domain
    result = np.fft.irfft(fft_s * S)
    # return left half (matches C: "return left half of real part")
    return result[:n]


# ── Main pileup kernel ────────────────────────────────────────────────────────

def apply_pileup_numpy(
    model_flux,          # ph/cm²/s/keV on internal 4096-point grid
    energ_lo,            # ARF bin lower edges [keV]
    energ_hi,            # ARF bin upper edges [keV]
    specresp,            # ARF effective area [cm²]
    exposure_time,       # total exposure [s]
    frame_time,          # per-frame time [s]  (EXPTIME keyword)
    fracexpo,            # fractional exposure (FRACEXPO from ARF)
    alpha,               # grade-migration parameter
    g0,                  # grade-zero probability
    num_regions,         # number of detection cells
    psf_frac,            # PSF fraction in pileup region
    max_num_terms=30,    # max pileup order
):
    """
    Pure NumPy implementation of Sherpa's apply_pileup (pileup.cc).

    Parameters
    ----------
    model_flux : array, shape (NUM_POINTS,)
        Differential photon flux [ph/cm²/s/keV] evaluated on the internal
        4096-point grid (_ENLO, _ENHI).  Caller must evaluate the
        astromodels source on this grid before calling.
    energ_lo, energ_hi : arrays
        ARF energy bin edges [keV].
    specresp : array
        ARF effective area [cm²].
    exposure_time : float  [s]
    frame_time : float     [s]
    fracexpo : float       [dimensionless]
    alpha, g0, num_regions, psf_frac : floats
    max_num_terms : int

    Returns
    -------
    results : array, shape (len(energ_lo),)
        Piled-up photon counts per detector bin [ph/frame * num_frames],
        ready to be folded through the RMF.
    pileup_fractions : array, shape (max_num_terms+1,)
    integral_ae : float
    num_terms : int
    """
    num_frames = exposure_time / frame_time
    num_bins   = len(energ_lo)

    # ── 1. Build arf_time on internal grid ────────────────────────────────
    arf_time = _build_arf_time(energ_lo, energ_hi, specresp, frame_time, fracexpo)

    # ── 2. arf_s = arf_time × model_flux  [ph/keV/frame] ─────────────────
    # Matches convert_spectrum(): arf_s[i] = arf[i] * s_den[i]
    arf_s = np.maximum(arf_time * model_flux, 0.0)

    # ── 3. perform_pileup ─────────────────────────────────────────────────
    pileup_fractions = np.zeros(max_num_terms + 1)

    # scale by psf_frac / (num_regions * fracexpo)
    scale = psf_frac / num_regions
    if fracexpo > 0:
        scale /= fracexpo

    arf_s_scaled = arf_s * scale
    integ_arf_s  = arf_s_scaled.sum()

    pileup_fractions[0] = 0.0
    pileup_fractions[1] = integ_arf_s

    if integ_arf_s == 0.0:
        return np.zeros(num_bins), pileup_fractions, 0.0, 0

    integral_ae = integ_arf_s / g0

    # Normalise to avoid float overflow (corrected at the end)
    arf_s_norm = arf_s_scaled / integ_arf_s

    # Pre-compute FFT of normalised array (zero-padded to 2*NUM_POINTS)
    padded = np.zeros(2 * NUM_POINTS)
    padded[NUM_POINTS:] = arf_s_norm
    fft_s = np.fft.rfft(padded)

    # Grade-migration coefficients: gfactors[i-2] = alpha^(i-1)
    alpha = abs(alpha)
    gfactors = np.array([alpha**(i - 1) for i in range(2, max_num_terms + 1)])

    results       = arf_s_scaled.copy()
    arf_s_tmp     = arf_s_norm.copy()
    i_factorial   = 1
    integ_arf_s_n = integ_arf_s
    total_prob    = 1.0 + integ_arf_s
    exp_factor    = np.exp(-integral_ae)
    num_terms     = max_num_terms

    for i in range(2, max_num_terms + 1):
        i_factorial   *= i
        integ_arf_s_n *= integ_arf_s
        norm_i         = integ_arf_s_n / i_factorial
        total_prob    += norm_i

        # Convolve arf_s_tmp with fft_s
        arf_s_tmp = _fft_convolve(fft_s, arf_s_tmp)

        norm_i *= gfactors[i - 2]
        results += norm_i * arf_s_tmp

        if i <= max_num_terms:
            pileup_fractions[i] = norm_i

        if total_prob * np.exp(-integ_arf_s) > MAX_PROBABILITY_CUTOFF:
            num_terms = i
            break

    # Apply frame/exposure correction (matches C: exp_factor *= num_frames * reg_size * fracexpo)
    exp_factor *= num_frames * num_regions * fracexpo
    results    *= exp_factor

    # Add unpiled fraction (photons that miss the pileup region)
    unpiled_frac = 1.0 - psf_frac
    if unpiled_frac > 0.0:
        results += arf_s * unpiled_frac * num_frames

    # ── 4. Rebin from internal grid to detector bins ──────────────────────
    final = _rebin_histogram(results, _ENLO, _ENHI, energ_lo, energ_hi)
    if final is None:
        final = np.zeros(num_bins)

    return final, pileup_fractions, integral_ae, num_terms


# ── Convenience: evaluate model on internal grid ──────────────────────────────

def eval_model_on_internal_grid(differential_flux_func):
    diff_edges = differential_flux_func(_E_EDGES)   # shape (4097,)
    diff_mid   = differential_flux_func(_E_MID)     # shape (4096,)
    de         = _ENHI - _ENLO
    integrated = de / 6.0 * (diff_edges[:-1] + 4.0 * diff_mid + diff_edges[1:])
    integrated[~np.isfinite(integrated)] = 0.0
    flux = integrated / de
    flux[~np.isfinite(flux)] = 0.0
    return flux