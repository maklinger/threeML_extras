"""
sed_propagator.py
-----------------
Posterior uncertainty propagation through a 3ML spectral model.
"""

import numpy as np
import astropy.units as u
from tqdm import tqdm
from threeML.random_variates import RandomVariates


def _evaluate_chunk(args):
    """
    Worker function for multiprocessing.Pool (fork context).

    Each worker receives a chunk of posterior samples and evaluates
    the model sequentially for each, reusing a single model instance.
    The model clone is accessed via a global set by the pool initializer.

    Parameters
    ----------
    args : tuple
        (worker_index, param_paths, param_matrix_chunk, Es_keV, efac)
        - worker_index       : int, indexes into _worker_clones
        - param_paths        : list of str, parameter paths
        - param_matrix_chunk : ndarray (chunk_size, n_params)
        - Es_keV             : ndarray (N_energies,)
        - efac               : ndarray (N_energies,)

    Returns
    -------
    ndarray, shape (chunk_size, N_energies)
    """
    worker_index, param_paths, param_matrix_chunk, Es_keV, efac = args
    wm = _worker_clones[worker_index]
    source_name = list(wm.sources.keys())[0]
    wm_source = wm.sources[source_name]

    chunk_size = len(param_matrix_chunk)
    results = np.empty((chunk_size, len(Es_keV)))

    for i, param_values in enumerate(param_matrix_chunk):
        for path, val in zip(param_paths, param_values):
            wm.parameters[path].value = val
        results[i] = efac * wm_source(Es_keV)

    return results


def _init_worker(clones):
    """Pool initializer: store clones in a global for worker access."""
    global _worker_clones
    _worker_clones = clones


class SEDPropagator:
    """
    Propagate posterior uncertainties through a 3ML spectral model
    and produce EFE (energy-flux) posterior samples at a grid of energies.

    The model is evaluated once per posterior sample with all energies
    evaluated simultaneously per sample.

    Parallelisation uses multiprocessing with the ``fork`` start method.
    Forked processes inherit the full parent memory, including dynamically
    created astromodels classes (e.g. from ``make_band_limited``), pybind11
    plugins, and template models — no serialisation or re-registration needed.

    Only ``n_jobs`` model instances are created — one per worker. Each worker
    processes its assigned chunk of posterior samples sequentially, reusing
    its single model instance.

    ``om`` is never evaluated directly. It is used only to read parameter
    values and metadata (free parameter paths, source name). All evaluations
    go through working model instances built by ``model_factory``.

    Because models with internal connections (e.g. ``am3ext3ml.set_bhjet``,
    ``model.link``) cannot be reliably cloned with ``astromodels.clone_model``,
    a user-supplied ``model_factory`` callable is used to construct fresh
    model instances. The factory receives the optimal model ``om`` and must
    return a fully wired ``astromodels.Model`` with parameter values copied
    from ``om``. If ``model_factory`` is None, ``astromodels.clone_model``
    is used as a fallback (suitable for simple, unconnected models).

    .. note::
        ``fork`` is the default on Linux and is safe in most scientific
        Python environments. It is not available on macOS (Python 3.8+
        defaults to ``spawn``) or Windows. On those platforms, fall back
        to ``n_jobs=1``.

    Parameters
    ----------
    om : astromodels.Model
        Model with the optimal (best-fit) parameters. Used for metadata
        and parameter values only — never evaluated directly.
    variates : dict
        Posterior variates keyed by parameter path in om.
    model_factory : callable or None
        Function with signature ``model_factory(om) -> astromodels.Model``.
        Must reconstruct the full model (all components, connections, links)
        and copy parameter values from ``om``. Called ``n_jobs`` times
        before forking (plus once for serial evaluation in ``get_optimal_model``
        and ``_propagate_serial``). If None, falls back to
        ``astromodels.clone_model``.

        Example::

            def make_model(om):
                starlight = TargetBlackBody()
                bhjet3ml = BHJetPlugin()
                bhjet3ml._setup(...)
                bhjet3ml.add_target(starlight, "starlight")
                am3ext3ml = AM3BHJetPerturbationPlugin()
                am3ext3ml.set_bhjet(bhjet3ml)
                # ... bounds, priors, free flags ...
                total = fdust * (bhjet3ml + starlight + am3ext3ml)
                ps = PointSource("M87", ra=0, dec=0, spectral_shape=total)
                model = Model(ps)
                model.link(starlight.lg_distance, bhjet3ml.lg_distance)
                model.link(starlight.redshift, bhjet3ml.redshift)
                for path, par in om.parameters.items():
                    if path not in model.parameters:
                        continue
                    if model.parameters[path].is_linked:
                        continue
                    model.parameters[path].value = par.value
                return model

    Nmax : int
        Maximum number of posterior samples to use. Default 1000.
    n_jobs : int
        Number of parallel worker processes. 1 = serial (default).
        -1 = use all available cores.
    cuts : dict
        Optional parameter range cuts. Keys are parameter paths (as they
        appear in ``variates``), values are ``[min, max]`` bounds.
        Samples outside the range on *any* parameter are discarded before
        subsampling to Nmax.
        Example: ``{"src.spectrum.main.BHJetPlugin.lg_jet_power_eddington": [-6, -4]}``
    """

    def __init__(
        self,
        om,
        variates,
        model_factory=None,
        Nmax: int = 1000,
        n_jobs: int = 1,
        cuts: dict = {},
    ) -> None:
        self.om = om
        self.n_jobs = n_jobs
        self.cuts = cuts
        self.source_name = list(om.sources.keys())[0]

        if model_factory is None:
            import astromodels
            self._model_factory = astromodels.clone_model
        else:
            self._model_factory = model_factory

        self.variates = variates
        self.Nmax = Nmax

        print("Optimal model parameters:")
        for name, par in self.om.free_parameters.items():
            print(f"  {par.name} : {par.value}")

        self.rvs = self._pick_samples()
        self.eflux_rvs = None

    # ------------------------------------------------------------------
    # Cut mask generation
    # ------------------------------------------------------------------

    def _generate_cuts(self) -> np.ndarray:
        """
        Build a boolean mask over all variates selecting samples that
        satisfy every range cut in ``self.cuts``.

        Returns
        -------
        mask : ndarray of bool, shape (N_samples,)

        Raises
        ------
        KeyError
            If a key in ``self.cuts`` is not found in ``self.variates``.
        ValueError
            If a cut value is not a two-element sequence.
        """
        first = next(iter(self.variates.values()))
        if not self.cuts:
            return np.ones(len(first), dtype=bool)

        mask = None
        for path, bounds in self.cuts.items():
            if path not in self.variates:
                raise KeyError(
                    f"Cut key '{path}' not found in variates. "
                    f"Available keys: {list(self.variates.keys())}"
                )
            bounds = np.asarray(bounds)
            if bounds.shape != (2,):
                raise ValueError(
                    f"Cut for '{path}' must be [min, max], got {bounds}."
                )
            xmin, xmax = bounds
            variate = np.asarray(self.variates[path])
            cut_mask = (variate >= xmin) & (variate <= xmax)
            mask = cut_mask if mask is None else (mask & cut_mask)

        n_total = len(first)
        n_kept = int(mask.sum())
        n_cut = n_total - n_kept
        if n_cut:
            print(
                f"Cuts removed {n_cut}/{n_total} samples "
                f"({100 * n_cut / n_total:.1f}%). {n_kept} samples remain."
            )
        return mask

    # ------------------------------------------------------------------
    # Sample selection
    # ------------------------------------------------------------------

    def _pick_samples(self) -> dict:
        """
        Select up to Nmax consistent posterior samples across all free
        parameters using the same random indices to preserve correlations.
        Samples that fail any cut in ``self.cuts`` are removed first.

        Returns
        -------
        dict mapping parameter path → samples array
        """
        cut_mask = self._generate_cuts()
        cut_indices = np.where(cut_mask)[0]

        if len(cut_indices) > self.Nmax:
            cut_indices = np.random.choice(cut_indices, size=self.Nmax, replace=False)

        arguments = {}
        for path in self.om.free_parameters:
            if self.source_name not in path:   # skip nuisance parameters
                continue
            arguments[path] = np.asarray(self.variates[path])[cut_indices]

        self.Nsamples = len(cut_indices)
        return arguments  # {path: samples_array}

    # ------------------------------------------------------------------
    # Energy conversion
    # ------------------------------------------------------------------

    def _efac(self, Es_keV: np.ndarray) -> np.ndarray:
        """E² factor: erg·keV so that efac * F(E)[1/keV/cm²/s] = EFE [erg/cm²/s]."""
        Es_erg = (Es_keV * u.keV).to(u.erg).value
        return Es_erg * Es_keV

    # ------------------------------------------------------------------
    # Propagation (serial)
    # ------------------------------------------------------------------

    def _propagate_serial(self, Es_keV: np.ndarray, efac: np.ndarray) -> np.ndarray:
        """Serial propagation using a single working model instance."""
        wm = self._model_factory(self.om)
        wm_source = wm.sources[self.source_name]
        samples = np.empty((self.Nsamples, len(Es_keV)))

        for i in tqdm(range(self.Nsamples), desc="Propagating"):
            for path, variates in self.rvs.items():
                wm.parameters[path].value = variates[i]
            samples[i] = efac * wm_source(Es_keV)
        return samples

    # ------------------------------------------------------------------
    # Propagation (parallel, fork)
    # ------------------------------------------------------------------

    def _propagate_parallel(self, Es_keV: np.ndarray, efac: np.ndarray) -> np.ndarray:
        """
        Parallel propagation using multiprocessing.Pool with fork.

        Only n_jobs model instances are created. Samples are split into
        n_jobs chunks; each worker processes its chunk sequentially on
        its own model instance.
        """
        import multiprocessing as mp
        import os

        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()

        param_paths = list(self.rvs.keys())
        param_matrix = np.column_stack(list(self.rvs.values()))
        # shape: (Nsamples, n_free_params)

        chunks = np.array_split(param_matrix, n_jobs)

        print(f"  Building {n_jobs} model instances ...")
        clones = [
            self._model_factory(self.om)
            for _ in tqdm(range(n_jobs), desc="Building models")
        ]

        args_list = [
            (i, param_paths, chunks[i], Es_keV, efac)
            for i in range(n_jobs)
        ]

        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(clones,),
        ) as pool:
            chunk_results = pool.map(_evaluate_chunk, args_list)

        # reassemble: list of (chunk_size, N_energies) → (Nsamples, N_energies)
        return np.vstack(chunk_results)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _propagate_spectrum(self, Es_keV: np.ndarray) -> None:
        """
        Evaluate EFE = E²·F(E) for each posterior sample and store as
        self.eflux_rvs: list of RandomVariates, one per energy point.
        """
        efac = self._efac(Es_keV)

        if self.n_jobs == 1:
            samples = self._propagate_serial(Es_keV, efac)
        else:
            print(f"Propagating in parallel with fork multiprocessing (n_jobs={self.n_jobs}) ...")
            samples = self._propagate_parallel(Es_keV, efac)

        self.eflux_rvs = [
            RandomVariates(samples[:, j]) for j in range(len(Es_keV))
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, Es, hpd: bool = True):
        """
        Propagate posterior samples and return credible interval bounds.

        Parameters
        ----------
        Es : array-like or astropy Quantity
            Energies at which to evaluate the model. Assumed keV if no unit.
        hpd : bool
            If True, return HPD interval. If False, return ETI. Default True.

        Returns
        -------
        interval : ndarray, shape (2, N_energies)
            Row 0: lower bound, Row 1: upper bound [erg/cm²/s].
        """
        Es_keV = Es.to("keV").value if isinstance(Es, u.Quantity) else np.asarray(Es)
        self._propagate_spectrum(Es_keV)
        return self._to_interval(self.eflux_rvs, hpd)

    def get_median(self) -> np.ndarray:
        """Posterior median EFE at each energy. Call propagate() first."""
        self._check_propagated()
        return np.array([v.median for v in self.eflux_rvs])

    def get_map(self) -> np.ndarray:
        """MAP EFE at each energy. Call propagate() first."""
        self._check_propagated()
        return np.array([float(v[np.argmax(v)]) for v in self.eflux_rvs])

    def get_optimal_model(self, Es) -> np.ndarray:
        """
        EFE at the optimal (best-fit) parameter values.

        Builds a fresh working model via model_factory to avoid mutating om.
        """
        Es_keV = Es.to("keV").value if isinstance(Es, u.Quantity) else np.asarray(Es)
        efac = self._efac(Es_keV)
        wm = self._model_factory(self.om)
        return efac * wm.sources[self.source_name](Es_keV)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_interval(self, eflux_rvs, hpd: bool) -> np.ndarray:
        if hpd:
            return np.array([
                v.highest_posterior_density_interval() for v in eflux_rvs
            ]).T
        else:
            return np.array([
                v.equal_tail_interval() for v in eflux_rvs
            ]).T

    def _check_propagated(self):
        if self.eflux_rvs is None:
            raise RuntimeError("Call propagate() before accessing results.")
