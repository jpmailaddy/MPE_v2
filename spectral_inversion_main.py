"""
POES MPE - Main Processing Module

This is the primary module that orchestrates the spectral inversion process for POES satellites.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize as scipy_minimize
import multiprocessing as mp
from functools import partial
import time
from spectral_functions import (RelativisticsMaxwellianElectron, RelativisticsMaxwellianProton,
                                PowerLaw, EnergyExponential,
                                DoubleRelativisticsMaxwellianElectron, DoubleRelativisticsMaxwellianProton)
from geometric_factors import get_electron_gf_array, get_proton_gf_array, logspace, logspace_midpoints
from penalty_functions import (PenaltyFunctionCalculator, first_guess_relmaxwell_electron,
                               first_guess_relmaxwell_proton, first_guess_powerlaw,
                               first_guess_exponential)
import os
from scipy.interpolate import interp1d


class POESSpectralInversion:
    """Main class for POES spectral inversion processing"""
    
    def __init__(self, min_energy=25., max_proton_energy=10000., max_electron_energy=10000.,
                 interp_level=27, dy=0.4, dt=16.0, 
                 optimization_tolerance=1e-3, max_optimization_iter=500):
        """
        Initialize POES spectral inversion processor
        
        Parameters
        ----------
        min_energy : float
            Minimum energy for spectral grid (keV)
        max_proton_energy : float
            Maximum proton energy (keV)
        max_electron_energy : float
            Maximum electron energy (keV)
        interp_level : int
            Number of interpolation points for spectral grid
        dy : float
            Relative error in measurements (0.4 = 40%)
        dt : float
            Integration time (seconds)
        optimization_tolerance : float
            Tolerance for spectral optimization convergence (default: 1e-3).
        max_optimization_iter : int
            Maximum iterations for spectral optimization (default: 500).
        """
        self.min_energy = min_energy
        self.max_proton_energy = max_proton_energy
        self.max_electron_energy = max_electron_energy
        self.interp_level = interp_level
        self.dy = dy
        self.dt = dt
        self.optimization_tolerance = optimization_tolerance
        self.max_optimization_iter = max_optimization_iter
        
        # Create logarithmic energy grid matching IDL readResponse:
        # edges = logspace(log10(minE), log10(maxE), interp+1)  [interp_level+1 edges]
        # energy = geometric-mean midpoints: sqrt(edges[i]*edges[i+1])  [interp_level points]
        # This ensures self.energy aligns with the IDL energy axis.
        self.energy = logspace_midpoints(np.log10(min_energy), np.log10(max_proton_energy), interp_level)
        
        # Geometric factors for POES instruments
        self.gf_proton = np.array([0.43, 1.35, 4.01, 11.29, 22.03])  # cm^2*sr*keV (P1-P5 bowtie)
        # default electron GF (fallback)
        self.gf_electron = np.array([1.12, 2.26, 2.44, 3.58])  # cm^2*sr*keV (E1-E4 bowtie)
        self.energy_proton = np.array([39., 115., 332., 1105., 2723.])  # keV
        self.energy_electron = np.array([72., 193., 419., 879.])  # keV
        
        # Spectral function instances
        self.spec_functions_electron = {
            'relmaxwell': RelativisticsMaxwellianElectron(),
            'powerlaw': PowerLaw(),
            'exponential': EnergyExponential(),
            'drelmaxwell': DoubleRelativisticsMaxwellianElectron(),
        }
        
        self.spec_functions_proton = {
            'relmaxwell': RelativisticsMaxwellianProton(),
            'powerlaw': PowerLaw(),
            'exponential': EnergyExponential(),
            'drelmaxwell': DoubleRelativisticsMaxwellianProton(),
        }
        # Attempt to read Electron->Electron response CSV and compute effective GF for E1-E4
        try:
            resp_struct = read_response_csv('Electron', 'Electron', self.min_energy, self.max_proton_energy, self.interp_level)
            resp = resp_struct['response']  # shape (n_energy, n_channels)
            deltaE = resp_struct['deltaE']
            # record path for provenance
            self.electron_electron_response_path = resp_struct.get('path')
            # Compute GF per channel: sum(response * deltaE) / 100 (matching IDL: response*deltaE then /100)
            gf_from_resp = np.sum(resp * deltaE[:, None], axis=0) / 100.0
            # Some response CSVs include a trailing "total" or summary column (last column)
            # with the summed geometric factor for all channels. Detect and drop it so we
            # don't accidentally treat the total as a physical channel (e.g. E4).
            if gf_from_resp.size >= 4:
                # If last column is approximately the sum of earlier columns, treat it as the total
                if np.isfinite(gf_from_resp[-1]) and np.isfinite(gf_from_resp[:-1]).all():
                    if np.isclose(gf_from_resp[-1], np.nansum(gf_from_resp[:-1]), rtol=1e-2, atol=1e-8):
                        # drop trailing total column
                        gf_true = gf_from_resp[:-1]
                    else:
                        gf_true = gf_from_resp
                else:
                    gf_true = gf_from_resp

                # Use available per-channel GFs. The processing pipeline commonly expects
                # E1-E4 (E4 may be a virtual channel derived from P6). If only 3 physical
                # electron columns are present (E1-E3), keep those and leave the E4 value
                # as the previously defined default (it will be filled in later via P6).
                if gf_true.size >= 4 and np.all(np.isfinite(gf_true[:4])):
                    self.gf_electron = gf_true[:4]
                elif gf_true.size >= 3 and np.all(np.isfinite(gf_true[:3])):
                    # set E1-E3 from response; leave E4 as previously defined default
                    self.gf_electron[:3] = gf_true[:3]
        except Exception:
            # keep default gf_electron if any issue reading CSV
            pass
        # Attempt to read Electron<-Proton response (how protons produce counts in electron telescopes)
        # This response will be used to estimate proton contamination in electron channels.
        self.K_ep = None
        try:
            resp_ep = read_response_csv('Electron', 'Proton', self.min_energy, self.max_proton_energy, self.interp_level)
            resp_mat = resp_ep['response']  # shape (n_energy, n_response_cols)
            deltaE_ep = resp_ep['deltaE']
            # record Electron<-Proton response path
            self.electron_proton_response_path = resp_ep.get('path')
            # Expect columns correspond to electron channels; ensure at least 3 (E1-E3)
            ncols = resp_mat.shape[1]
            # Some CSVs include a trailing total/sum column. If the last column is
            # approximately the sum of the other columns on a per-row basis, drop it
            # before deciding how many channel columns we have.
            if ncols >= 2:
                # check per-row if last column equals sum of others (allow small rounding)
                try:
                    row_sum = np.nansum(resp_mat[:, :-1], axis=1)
                    if np.allclose(resp_mat[:, -1], row_sum, rtol=1e-2, atol=1e-8):
                        # drop trailing total column
                        resp_mat = resp_mat[:, :-1]
                        ncols = resp_mat.shape[1]
                except Exception:
                    # conservative: if check fails, keep original resp_mat
                    pass

            if ncols < 3:
                # Not enough columns to estimate contamination
                self.K_ep = None
            else:
                # Build K_ep = response * deltaE * dt ; shape (n_energy, n_electron_channels)
                # Use first 4 columns if available, but contamination concerns E1-E3 primarily
                use_cols = min(ncols, 4)
                resp_used = resp_mat[:, :use_cols]
                # IDL semantics: response*deltaE then /100, then *dt
                self.K_ep = resp_used * deltaE_ep[:, None] * self.dt / 100.0
        except Exception:
            self.K_ep = None
    
    def _forward_model_counts(self, q, spectral_model, K_matrix):
        """Compute predicted counts via forward model (IDL-equivalent).
        
        Uses simple matrix multiplication: lambda = K.T @ flux
        This matches IDL: cts = Kwfp##f where Kwfp = response * deltaE / 100 * dt
        
        K_matrix shape: (n_energy, n_channels).  Returns shape (n_channels,).
        """
        spec_func = self.spec_functions_proton[spectral_model]
        try:
            flux = spec_func(q, self.energy)
        except Exception:
            return None
        if not np.all(np.isfinite(flux)):
            return None
        
        # IDL-equivalent forward model: lambda = K @ f
        # K_matrix already encodes: response * deltaE / 100 * dt
        # So lambda[j] = sum_i(K_matrix[i,j] * flux[i])
        # Use transpose for matrix multiplication: (n_channels, n_energy) @ (n_energy,) -> (n_channels,)
        lambda_pred = K_matrix.T @ flux
        
        return lambda_pred

    def _penalty_poisson_gaussian(self, y_counts, lambda_pred, dy):
        """IDL-equivalent per-channel penalty (Poisson or Gaussian).

        Matches IDL ``penalty_function_magpd``:
            if y < dy^{-2}:  ell = lambda - y*log(lambda)          (Poisson)
            else:             ell = 0.5*((log(y)-log(lambda))/dy)^2  (Gaussian)

        Parameters
        ----------
        y_counts : array
            Observed counts (not count rates – multiply count rate by dt before passing).
        lambda_pred : array
            Predicted counts from forward model.
        dy : float
            Relative measurement error (e.g. 0.4).

        Returns
        -------
        tuple (total_ell, ell_per_channel)
        """
        dy_sq_inv = dy ** (-2)
        ell = np.zeros(len(y_counts))
        for i in range(len(y_counts)):
            lam = lambda_pred[i]
            yi = y_counts[i]
            if lam <= 0 or not np.isfinite(lam):
                ell[i] = 1e10
                continue
            if yi < dy_sq_inv:  # Poisson statistics
                # yi may be 0; log(lam) is safe because lam > 0
                yi_safe = max(yi, 0.0)
                if yi_safe > 0:
                    ell[i] = lam - yi_safe * np.log(lam)
                else:
                    ell[i] = lam
            else:  # Gaussian statistics
                yi_safe = max(yi, 1e-30)
                ell[i] = 0.5 * ((np.log(yi_safe) - np.log(lam)) / dy) ** 2
        total = np.sum(ell)
        if not np.isfinite(total):
            total = 1e10
        return total, ell

    def _penalty_proton_spectrum(self, q, proton_counts_dt, spectral_model):
        """
        Calculate IDL-equivalent penalty function for proton spectrum fitting.

        Uses the same Poisson/Gaussian log-likelihood as IDL ``penalty_function_magpd``.
        ``proton_counts_dt`` should be counts (count_rate * dt), matching IDL ``y = protoncount[0:4,i]*dt``.

        Parameters
        ----------
        q : array
            Spectral parameters
        proton_counts_dt : array
            Observed proton counts (count_rate * dt) for P1-P5  (shape (5,))
        spectral_model : str
            Name of spectral model ('relmaxwell', 'powerlaw', 'exponential', 'drelmaxwell')

        Returns
        -------
        float
            Total penalty value (sum of per-channel likelihoods)
        """
        if hasattr(self, 'K_proton') and self.K_proton is not None:
            K = self.K_proton[:, :5]  # Only P1-P5 for fitting
            lambda_pred = self._forward_model_counts(q, spectral_model, K)
            if lambda_pred is None or np.any(lambda_pred <= 0):
                return 1e10
        else:
            # Fallback: simple GF-based forward model (counts = GF * integral(flux * dE))
            spec_func = self.spec_functions_proton[spectral_model]
            try:
                flux = spec_func(q, self.energy)
            except Exception:
                return 1e10
            lambda_pred = np.zeros(len(self.gf_proton))
            for i, gf_val in enumerate(self.gf_proton):
                lambda_pred[i] = trapezoid(flux * gf_val, self.energy) * self.dt
            if np.any(lambda_pred <= 0):
                return 1e10

        total_ell, _ = self._penalty_poisson_gaussian(proton_counts_dt, lambda_pred, self.dy)
        return total_ell

    def _penalty_electron_spectrum(self, q, electron_counts_dt, spectral_model):
        """
        IDL-equivalent penalty for electron spectrum fitting.

        Mirrors ``penalty_function_maged`` (same Poisson/Gaussian structure as magpd).

        Parameters
        ----------
        q : array
            Spectral parameters
        electron_counts_dt : array
            Observed electron counts (count_rate * dt) for E1-E4  (shape (4,))
        spectral_model : str
            Name of spectral model

        Returns
        -------
        float
            Total penalty value
        """
        if hasattr(self, 'K_electron') and self.K_electron is not None:
            K = self.K_electron
            spec_func = self.spec_functions_electron[spectral_model]
            try:
                flux = spec_func(q, self.energy)
            except Exception:
                return 1e10
            if not np.all(np.isfinite(flux)):
                return 1e10
            
            # IDL-equivalent forward model: lambda = K.T @ flux
            # K already encodes: response * deltaE / 100 * dt
            lambda_pred = K.T @ flux
        else:
            # Fallback
            spec_func = self.spec_functions_electron[spectral_model]
            try:
                flux = spec_func(q, self.energy)
            except Exception:
                return 1e10
            lambda_pred = np.zeros(len(self.gf_electron))
            for i, gf_val in enumerate(self.gf_electron):
                lambda_pred[i] = trapezoid(flux * gf_val, self.energy) * self.dt
        if np.any(lambda_pred <= 0):
            return 1e10
        total_ell, _ = self._penalty_poisson_gaussian(electron_counts_dt, lambda_pred, self.dy)
        return total_ell
    
    def fit_proton_spectrum(self, proton_counts):
        """
        Fit proton spectrum to P1-P5 channels with optimization.

        Matches IDL ``spectral_fits_00.pro``:
          - ``y = protoncount[0:4, i] * dt``  (counts, not count rates)
          - Minimize IDL-equivalent Poisson/Gaussian log-likelihood penalty
          - Record ``ell`` for each model to compute IDL-style model weights:
              ``w = exp(-ell - n_params)``  (2 for 2-param models, 4 for DM)

        Parameters
        ----------
        proton_counts : array
            Proton count rates (P1-P5), shape (5,)  [counts/sec]

        Returns
        -------
        dict
            Per-model dict with keys 'q0', 'q', 'ell', 'converged'.
            'ell' is the total penalty at the optimal q (used for model weighting).
        """
        # IDL: y = protoncount[0:4,i]*dt  — convert count rates to counts
        y_counts = np.asarray(proton_counts, dtype=float) * self.dt

        # --- Fast-path: if all channels have essentially no detected counts, return
        # sentinel results (mirrors IDL where amoeba returns scalar 1 and Q is set to -1e31).
        # Expressed in raw counts (y_counts = rate*dt), not rate, so the floor means the
        # same thing regardless of integration time. 2 counts matches the original IDL
        # noise floor of 0.125 counts/sec at dt=16s (0.125*16=2); using <= (not <) also
        # catches inputs sitting exactly on the floor, which previously fell through to
        # the optimizer.
        _NOISE_FLOOR_COUNTS = 2.0
        if np.all(y_counts <= _NOISE_FLOOR_COUNTS):
            # For counts below noise floor, return a very small flux
            # Use q values that produce ~0 flux: exp(-50) ensures negligible values
            sentinel_q = np.array([-50., 0.])
            sentinel = {'q0': sentinel_q, 'q': sentinel_q,
                        'ell': 1e10, 'converged': False}
            sentinel4 = {'q0': np.array([-50., 0., -50., 0.]), 'q': np.array([-50., 0., -50., 0.]),
                         'ell': 1e10, 'converged': False}
            return {'relmaxwell': sentinel.copy(), 'powerlaw': sentinel.copy(),
                    'exponential': sentinel.copy(), 'drelmaxwell': sentinel4}

        # Initial flux estimate (count rate / GF)
        jflux = proton_counts / self.gf_proton

        results = {}

        def _optimize(model_name, q0, n_params):
            """Optimize a single model and return (q_opt, ell_opt, converged)."""
            ell0 = self._penalty_proton_spectrum(q0, y_counts, model_name)
            try:
                opt = scipy_minimize(
                    self._penalty_proton_spectrum,
                    q0,
                    args=(y_counts, model_name),
                    method='Nelder-Mead',
                    options={'xatol': self.optimization_tolerance, 
                             'fatol': self.optimization_tolerance, 
                             'maxiter': self.max_optimization_iter}
                )
                if opt.fun < ell0:
                    return opt.x, opt.fun, opt.success
                else:
                    return q0, ell0, False
            except Exception:
                return q0, ell0, False

        # Relativistic Maxwellian (2 params)
        q0_rm = first_guess_relmaxwell_proton(jflux, self.energy_proton)
        q_rm, ell_rm, conv_rm = _optimize('relmaxwell', q0_rm, 2)
        results['relmaxwell'] = {'q0': q0_rm, 'q': q_rm, 'ell': ell_rm, 'converged': conv_rm}

        # Power Law (2 params)
        q0_pl = first_guess_powerlaw(jflux, self.energy_proton)
        q_pl, ell_pl, conv_pl = _optimize('powerlaw', q0_pl, 2)
        results['powerlaw'] = {'q0': q0_pl, 'q': q_pl, 'ell': ell_pl, 'converged': conv_pl}

        # Energy Exponential (2 params)
        q0_ee = first_guess_exponential(jflux, self.energy_proton)
        q_ee, ell_ee, conv_ee = _optimize('exponential', q0_ee, 2)
        results['exponential'] = {'q0': q0_ee, 'q': q_ee, 'ell': ell_ee, 'converged': conv_ee}

        # Double Relativistic Maxwellian (4 params)
        q0_dm = np.concatenate([q0_rm, q0_rm])
        q_dm, ell_dm, conv_dm = _optimize('drelmaxwell', q0_dm, 4)
        results['drelmaxwell'] = {'q0': q0_dm, 'q': q_dm, 'ell': ell_dm, 'converged': conv_dm}

        return results
    
    def fit_electron_spectrum(self, electron_counts_corrected):
        """
        Fit electron spectrum to E1-E4 channels.

        Mirrors ``fit_proton_spectrum`` but for electrons, using the electron
        spectral functions and (if available) the ``K_electron`` weighting matrix.

        Parameters
        ----------
        electron_counts_corrected : array
            Corrected electron count rates (E1-E4), shape (4,)  [counts/sec]

        Returns
        -------
        dict
            Per-model dict with keys 'q0', 'q', 'ell', 'converged'.
        """
        # IDL: y = new_electronCount[*,i]*dt  — convert count rates to counts
        y_counts = np.asarray(electron_counts_corrected, dtype=float) * self.dt

        # --- Fast-path: if all corrected counts are zero, skip optimization
        # Use q values that produce ~0 flux (exp(-50) is negligible), mirroring the
        # proton noise-floor sentinel. q=[0,0] previously left RelMaxwell/DblRelMaxwell
        # unsuppressed (exp(0)=1) and PowerLaw/Exponential flat at 1, producing a large,
        # identical, spuriously energy-increasing spectrum for every all-zero observation.
        if np.all(np.asarray(electron_counts_corrected, dtype=float) == 0.0):
            sentinel_q = np.array([-50., 0.])
            sentinel = {'q0': sentinel_q, 'q': sentinel_q,
                        'ell': 1e10, 'converged': False}
            sentinel4 = {'q0': np.array([-50., 0., -50., 0.]), 'q': np.array([-50., 0., -50., 0.]),
                         'ell': 1e10, 'converged': False}
            return {'relmaxwell': sentinel.copy(), 'powerlaw': sentinel.copy(),
                    'exponential': sentinel.copy(), 'drelmaxwell': sentinel4}

        # Initial flux estimate
        jflux = electron_counts_corrected / self.gf_electron

        results = {}

        def _optimize(model_name, q0):
            ell0 = self._penalty_electron_spectrum(q0, y_counts, model_name)
            try:
                opt = scipy_minimize(
                    self._penalty_electron_spectrum,
                    q0,
                    args=(y_counts, model_name),
                    method='Nelder-Mead',
                    options={'xatol': self.optimization_tolerance, 
                             'fatol': self.optimization_tolerance, 
                             'maxiter': self.max_optimization_iter}
                )
                if opt.fun < ell0:
                    return opt.x, opt.fun, opt.success
                else:
                    return q0, ell0, False
            except Exception:
                return q0, ell0, False

        # Relativistic Maxwellian (2 params)
        q0_rm = first_guess_relmaxwell_electron(jflux, self.energy_electron)
        q_rm, ell_rm, conv_rm = _optimize('relmaxwell', q0_rm)
        results['relmaxwell'] = {'q0': q0_rm, 'q': q_rm, 'ell': ell_rm, 'converged': conv_rm}

        # Power Law (2 params)
        q0_pl = first_guess_powerlaw(jflux, self.energy_electron)
        q_pl, ell_pl, conv_pl = _optimize('powerlaw', q0_pl)
        results['powerlaw'] = {'q0': q0_pl, 'q': q_pl, 'ell': ell_pl, 'converged': conv_pl}

        # Energy Exponential (2 params)
        q0_ee = first_guess_exponential(jflux, self.energy_electron)
        q_ee, ell_ee, conv_ee = _optimize('exponential', q0_ee)
        results['exponential'] = {'q0': q0_ee, 'q': q_ee, 'ell': ell_ee, 'converged': conv_ee}

        # Double Relativistic Maxwellian (4 params)
        q0_dm = np.concatenate([q0_rm, q0_rm])
        q_dm, ell_dm, conv_dm = _optimize('drelmaxwell', q0_dm)
        results['drelmaxwell'] = {'q0': q0_dm, 'q': q_dm, 'ell': ell_dm, 'converged': conv_dm}

        return results
    
    def remove_proton_contamination(self, proton_flux, electron_counts, 
                                    e1_proton_range=(200, 2600),
                                    e2_proton_range=(250, 2600),
                                    e3_proton_range=(500, 2600)):
        """
        Remove proton contamination from electron channels
        
        Parameters
        ----------
        proton_flux : array
            Proton differential flux (1/cm^2/sr/keV/s)
        electron_counts : array
            Original electron counts (E1-E3), shape (3,)
        e1_proton_range : tuple
            Energy range (keV) of proton contamination in E1
        e2_proton_range : tuple
            Energy range (keV) of proton contamination in E2
        e3_proton_range : tuple
            Energy range (keV) of proton contamination in E3
        
        Returns
        -------
        tuple
            corrected_counts (array): Corrected electron counts (length 3)
            proton_contamination (array): Estimated proton contamination for E1-E3 (length 3)
        """
        # If we have a precomputed K_ep matrix (n_energy x n_electron_channels), use it
        if self.K_ep is not None:
            # proton_flux expected shape (n_energy,)
            # p_lambda = K_ep.T @ proton_flux  -> expected counts per electron channel (counts)
            # Use IDL-equivalent matrix multiplication (K_ep already encodes deltaE and dt)
            try:
                p_lambda = self.K_ep.T @ proton_flux
            except Exception:
                # Dimension mismatch fallback: interpolate or raise
                raise RuntimeError('Dimension mismatch between K_ep and proton_flux in remove_proton_contamination')

            # Convert to count rates (counts/sec) as in IDL: pContam = K3 ## flux ; protoncontamination = pContam/dt
            proton_contamination_rate = p_lambda / self.dt

            # Subtract contamination rates from provided electron_counts (assumed counts/sec)
            # electron_counts expected length 3 (E1-E3)
            corrected_counts = np.array(electron_counts, dtype=float)
            # Only subtract for available channels
            n_ch = min(len(corrected_counts), len(proton_contamination_rate))
            corrected_counts[:n_ch] = corrected_counts[:n_ch] - proton_contamination_rate[:n_ch]

            # Clip negatives
            corrected_counts[corrected_counts < 0] = 0.0

            # Return corrected counts for E1-E3 and contamination rates for E1-E3
            return corrected_counts, proton_contamination_rate[:n_ch]

        # Fallback (legacy): integrate proton_flux over given energy ranges (keeps previous behavior)
        idx_e1 = np.where((self.energy >= e1_proton_range[0]) & (self.energy <= e1_proton_range[1]))[0]
        idx_e2 = np.where((self.energy >= e2_proton_range[0]) & (self.energy <= e2_proton_range[1]))[0]
        idx_e3 = np.where((self.energy >= e3_proton_range[0]) & (self.energy <= e3_proton_range[1]))[0]

        proton_contam_e1 = trapezoid(proton_flux[idx_e1], self.energy[idx_e1]) if idx_e1.size > 1 else 0.0
        proton_contam_e2 = trapezoid(proton_flux[idx_e2], self.energy[idx_e2]) if idx_e2.size > 1 else 0.0
        proton_contam_e3 = trapezoid(proton_flux[idx_e3], self.energy[idx_e3]) if idx_e3.size > 1 else 0.0

        corrected_counts = np.array([
            electron_counts[0] - proton_contam_e1,
            electron_counts[1] - proton_contam_e2,
            electron_counts[2] - proton_contam_e3,
        ])
        proton_contamination = np.array([proton_contam_e1, proton_contam_e2, proton_contam_e3])
        corrected_counts[corrected_counts < 0] = 0.
        return corrected_counts, proton_contamination
    
    def get_proton_flux_from_fit(self, q, spectral_model='relmaxwell'):
        """
        Calculate proton flux from fitted parameters
        
        Parameters
        ----------
        q : array
            Fitted spectral parameters
        spectral_model : str
            Name of spectral model
        
        Returns
        -------
        array
            Proton flux values at energy grid points
        """
        spec_func = self.spec_functions_proton[spectral_model]
        flux = spec_func(q, self.energy)
        
        return flux
    
    def get_electron_flux_from_fit(self, q, spectral_model='relmaxwell'):
        """
        Calculate electron flux from fitted parameters
        
        Parameters
        ----------
        q : array
            Fitted spectral parameters
        spectral_model : str
            Name of spectral model
        
        Returns
        -------
        array
            Electron flux values at energy grid points
        """
        spec_func = self.spec_functions_electron[spectral_model]
        flux = spec_func(q, self.energy)
        
        return flux

    def get_weighted_proton_flux(self, fit_results):
        """Compute IDL-style log-weighted combined proton flux from all 4 spectral models.

        Matches IDL ``plot_fit_pflux``:
            wRM = exp(-ell_RM - 2)   (2-param model penalty)
            wDM = exp(-ell_DM - 4)   (4-param model penalty)
            wPL = exp(-ell_PL - 2)
            wEE = exp(-ell_EE - 2)
            w_norm = [wRM, wPL, wEE, wDM] / sum(...)
            ln_combined = sum(w_norm * log(flux_model))
            combined_flux = exp(ln_combined)

        If a model did not converge (ell is very large), its weight becomes ~0.

        Parameters
        ----------
        fit_results : dict
            Output of :meth:`fit_proton_spectrum`.

        Returns
        -------
        tuple (combined_flux, weights)
            combined_flux : array, shape (n_energy,)
            weights : array, shape (4,)  [wRM, wPL, wEE, wDM] normalised
        """
        # Per-model (ell, n_free_params) pairs following IDL order
        model_info = [
            ('relmaxwell',  2),
            ('powerlaw',    2),
            ('exponential', 2),
            ('drelmaxwell', 4),
        ]
        fluxes = []
        raw_weights = []
        for model_name, n_params in model_info:
            res = fit_results.get(model_name)
            if res is None:
                raw_weights.append(0.0)
                fluxes.append(None)
                continue
            ell = res.get('ell', 1e10)
            # Clamp ell+n_params so exp argument stays above ~-500 (avoids underflow to 0 silently)
            ell_clamped = min(float(ell), 500.0) if np.isfinite(ell) else 500.0
            w = np.exp(-ell_clamped - float(n_params))
            raw_weights.append(w)
            flux = self.get_proton_flux_from_fit(res['q'], model_name)
            fluxes.append(flux)

        total_w = sum(raw_weights)
        if total_w <= 0:
            # Fallback: use RM only
            flux_rm = self.get_proton_flux_from_fit(fit_results['relmaxwell']['q'], 'relmaxwell')
            return flux_rm, np.array([1.0, 0.0, 0.0, 0.0])

        norm_weights = np.array(raw_weights) / total_w

        # Log-weighted combination (IDL: ln_combined = sum(w * log(flux)))
        ln_combined = np.zeros(len(self.energy))
        tiny = 1e-31
        for i, (model_name, _) in enumerate(model_info):
            if fluxes[i] is None:
                continue
            f = np.where(fluxes[i] > 0, fluxes[i], tiny)
            ln_combined += norm_weights[i] * np.log(f)

        combined_flux = np.exp(ln_combined)
        return combined_flux, norm_weights

    def get_weighted_electron_flux(self, fit_results):
        """Compute IDL-style log-weighted combined electron flux from all 4 spectral models.

        Mirrors :method:`get_weighted_proton_flux` for the electron telescope.
      
        Parameters
        ----------
        fit_results : dict
            Output of :meth:`fit_electron_spectrum`.

        Returns
        -------
        tuple (combined_flux, weights)
            combined_flux : array, shape (n_energy,)
            weights : array, shape (4,)  [wRM, wPL, wEE, wDM] normalised
        """
        model_info = [
            ('relmaxwell',  2),
            ('powerlaw',    2),
            ('exponential', 2),
            ('drelmaxwell', 4),
        ]
        fluxes = []
        raw_weights = []
        for model_name, n_params in model_info:
            res = fit_results.get(model_name)
            if res is None:
                raw_weights.append(0.0)
                fluxes.append(None)
                continue
            ell = res.get('ell', 1e10)
            ell_clamped = min(float(ell), 500.0) if np.isfinite(ell) else 500.0
            w = np.exp(-ell_clamped - float(n_params))
            raw_weights.append(w)
            flux = self.get_electron_flux_from_fit(res['q'], model_name)
            fluxes.append(flux)

        total_w = sum(raw_weights)
        if total_w <= 0:
            flux_rm = self.get_electron_flux_from_fit(fit_results['relmaxwell']['q'], 'relmaxwell')
            return flux_rm, np.array([1.0, 0.0, 0.0, 0.0])

        norm_weights = np.array(raw_weights) / total_w

        ln_combined = np.zeros(len(self.energy))
        tiny = 1e-31
        for i, (model_name, _) in enumerate(model_info):
            if fluxes[i] is None:
                continue
            f = np.where(fluxes[i] > 0, fluxes[i], tiny)
            ln_combined += norm_weights[i] * np.log(f)

        combined_flux = np.exp(ln_combined)
        return combined_flux, norm_weights


def _find_response_csv(telescope, response_type):
    """Find the best-matching response CSV in the repo for the given telescope/response_type.
    Prefers files with suffix _v2 if present.
    Returns full path or raises FileNotFoundError.
    """
    base = f"POES_{telescope}Telescope_{response_type}Response"
    cwd = os.path.dirname(__file__)
    candidates = []
    for suffix in ['', '_v2', '.csv', '_v2.csv']:
        # try combinations
        for ext in ['.csv', '']:
            name = base + (suffix if suffix.endswith('.csv') else suffix) 
            # ensure it ends with .csv
            if not name.endswith('.csv'):
                name = name + '.csv'
            path = os.path.join(cwd, name)
            if os.path.isfile(path):
                candidates.append(path)
    # fallback: search for any file starting with base
    if not candidates:
        for fn in os.listdir(cwd):
            if fn.startswith(base) and fn.endswith('.csv'):
                candidates.append(os.path.join(cwd, fn))
    if not candidates:
        raise FileNotFoundError(f"No response CSV found for {telescope}/{response_type} in {cwd}")
    
    # Prefer files with _v2 suffix first (as documented), then fall back to most columns
    v2_candidates = [c for c in candidates if '_v2' in os.path.basename(c)]
    non_v2_candidates = [c for c in candidates if '_v2' not in os.path.basename(c)]
    
    if v2_candidates:
        # If we have _v2 files, prefer the one with the most columns among them
        best = None
        best_cols = -1
        for c in v2_candidates:
            try:
                data = np.loadtxt(c, delimiter=',')
                # ensure 2D
                if data.ndim == 1:
                    cols = max(0, data.size - 1)
                else:
                    cols = data.shape[1] - 1
            except Exception:
                cols = -1
            if cols > best_cols:
                best_cols = cols
                best = c
        if best is not None:
            return best
    
    # Fall back to non-_v2 files if no _v2 files found or they were not usable
    if non_v2_candidates:
        best = None
        best_cols = -1
        for c in non_v2_candidates:
            try:
                data = np.loadtxt(c, delimiter=',')
                # ensure 2D
                if data.ndim == 1:
                    cols = max(0, data.size - 1)
                else:
                    cols = data.shape[1] - 1
            except Exception:
                cols = -1
            if cols > best_cols:
                best_cols = cols
                best = c
        if best is not None:
            return best
    
    raise FileNotFoundError(f"No usable response CSV found for {telescope}/{response_type} in {cwd}")


def read_response_csv(telescope, response_type, min_energy, max_proton_energy, interp_level):
    """Read response CSV and return dict like IDL readResponse: {'response': resp, 'energy': energy, 'deltaE': deltaE}

    CSV expected format: first column = energy midpoints, subsequent columns = response per channel.
    """
    path = _find_response_csv(telescope, response_type)
    data = np.loadtxt(path, delimiter=',')
    # first column is energy midpoints
    energy_csv = data[:, 0].astype(float)
    response = data[:, 1:]

    # compute edges and deltaE consistent with IDL readResponse:
    #   edges = logspace(log10(minE), log10(maxE), interp+1)  [interp_level+1 edges]
    #   deltaE[i] = edges[i+1] - edges[i]
    edges = logspace(np.log10(min_energy), np.log10(max_proton_energy), interp_level + 1)
    deltaE = edges[1:] - edges[:-1]

    # Interpolate response to processor energy grid.
    # IDL uses geometric-mean midpoints: midpoints[i] = sqrt(edges[i]*edges[i+1])
    target_midpoints = logspace_midpoints(np.log10(min_energy), np.log10(max_proton_energy), interp_level)
    # if csv energy matches target, skip interpolation
    if energy_csv.shape[0] != target_midpoints.shape[0] or not np.allclose(energy_csv, target_midpoints):
        # Interpolate each response column in log-response vs log-energy space to
        # match the IDL implementation which interpolates alog10(response) and
        # then exponentiates (response = 10.^response in IDL).
        resp_interp = np.zeros((len(target_midpoints), response.shape[1]))
        tiny = 1e-18
        for j in range(response.shape[1]):
            col = response[:, j]
            # mask finite entries
            mask = np.isfinite(col) & np.isfinite(energy_csv)
            if mask.sum() < 2:
                resp_interp[:, j] = np.nan
                continue
            # replace zeros or negative with a tiny positive before taking log10
            col_masked = col[mask].copy()
            col_masked = np.where(col_masked <= 0, tiny, col_masked)
            # interpolate log10(response) as a function of log10(energy)
            interp_log = interp1d(np.log10(energy_csv[mask]), np.log10(col_masked),
                                  bounds_error=False, fill_value=np.log10(tiny))
            resp_interp[:, j] = 10.0 ** (interp_log(np.log10(target_midpoints)))
        response = resp_interp
        energy = target_midpoints
    else:
        energy = energy_csv

    return {'response': response, 'energy': energy, 'deltaE': deltaE, 'path': path}


def _apply_electron_quality_checks(new_electron_count):
    """Apply IDL-equivalent quality checks to the corrected 4-channel electron count array.

    Mirrors the three ``count_rate_issues`` filter passes in IDL ``spectral_fits_00.pro``:

    Pass 1 – If E1 > 2.5 AND (E3 == 0 OR E3 < E4) AND E2 != 0:
                 If E4 != 0 → zero out all 4 channels (IDL chose to zero rather than
                 interpolate in this branch).

    Pass 2 – If E1 > 2.5 AND E2 < E4 → zero out all 4 channels.

    Pass 3 – If E1 > 2.5 AND E1 < E4 → zero out all 4 channels.

    Also zeros any negative values (equivalent to IDL ``badData = where(new_electroncount lt 0.)``).

    Parameters
    ----------
    new_electron_count : ndarray, shape (4,)
        Corrected electron count rates [E1, E2, E3, E4].  Modified in-place.

    Returns
    -------
    ndarray  – the (possibly modified) array (same object as input).
    """
    # Zero negatives first (IDL: badData = where(new_electroncount lt 0.))
    new_electron_count[new_electron_count < 0.0] = 0.0

    E1, E2, E3, E4 = new_electron_count[0], new_electron_count[1], new_electron_count[2], new_electron_count[3]

    # Pass 1: E1 > 2.5 AND (E3 == 0 OR E3 < E4) AND E2 != 0 AND E4 != 0 → zero all
    if E1 > 2.5:
        if (E3 == 0.0 or E3 < E4) and E2 != 0.0:
            if E4 != 0.0:
                new_electron_count[:] = 0.0
                return new_electron_count

    # Re-read after potential modification above
    E1, E2, E3, E4 = new_electron_count[0], new_electron_count[1], new_electron_count[2], new_electron_count[3]

    # Pass 2: E1 > 2.5 AND E2 < E4 → zero all
    if E1 > 2.5 and E2 < E4:
        new_electron_count[:] = 0.0
        return new_electron_count

    # Re-read after potential modification
    E1, E2, E3, E4 = new_electron_count[0], new_electron_count[1], new_electron_count[2], new_electron_count[3]

    # Pass 3: E1 > 2.5 AND E1 < E4 → zero all
    if E1 > 2.5 and E1 < E4:
        new_electron_count[:] = 0.0

    return new_electron_count


def process_poes_data(proton_counts, electron_counts, date_str='2019-09-06',
                     satellite='m01', output_file=None, p6_scale=1.0,
                     electron_counts_90deg=None, apply_angular_correction=False,
                     latitude=None, time_array=None,
                     n_workers=1, use_caching=False, dt=16.0,
                     optimization_tolerance=1e-3, max_optimization_iter=500):
    """Main processing function for POES data.

    Processing pipeline matches IDL ``spectral_fits_00.pro``:

    1. Build K matrices from response CSVs:
       - ``K_proton`` (K1) : ProtonTelescope × ProtonResponse  – P1-P6 (n_energy × 6)
       - ``K3``            : ElectronTelescope × ProtonResponse – E1-E3 (n_energy × 3)
       - ``K4``            : ElectronTelescope × ElectronResponse – E1-E3 (n_energy × 3)
       - ``K2_p6``         : ProtonTelescope × ElectronResponse, P6 column – for E4 (n_energy × 1)
       - ``K_electron`` (Kwfe) = [[K4], [K2_p6]] (n_energy × 4)

    2. Per time step:
       a. Fit proton spectrum (P1-P5) → weighted combined flux
       b. Compute pLambda = K_proton^T @ p_flux  (P1-P6 predicted counts)
       c. Subtract proton contamination from E1-E3:
              protoncontamination = (K3^T @ p_flux) / dt
              new_E1-E3 = electronCount - protoncontamination
       d. Build virtual E4 = measured P6 rate − calculated P6 rate
          (= protonCount[5] − pLambda[5]/dt)
       e. Apply IDL quality checks to [E1, E2, E3, E4]
       f. Fit electron spectrum (E1-E4) → weighted combined flux

    Parameters
    ----------
    proton_counts : ndarray, shape (6, n_time)
        Raw proton count rates [P1-P6] × time  [counts/sec].
    electron_counts : ndarray, shape (3, n_time)
        Raw electron count rates [E1-E3] × time  [counts/sec].
    date_str : str
    satellite : str
    output_file : str or None
    p6_scale : float
        Scale factor applied to the P5 GF when approximating P6 in the
        fallback (no CSV) path.
    electron_counts_90deg : ndarray, shape (3, n_time), optional
        Electron count rates from 90° telescope for angular correction.
    apply_angular_correction : bool
        Whether to apply Selesnick angular correction (requires electron_counts_90deg).
    latitude : array, optional
        Geographic latitude for angular correction.
    time_array : array of datetime, optional
        Time points for angular correction.
    n_workers : int
        Number of parallel workers for processing (1 = sequential, >1 = parallel).
    use_caching : bool
        Enable caching of spectral fits for similar count patterns.
    dt : float
        Integration time (seconds) matching the resolution of ``proton_counts``/
        ``electron_counts`` (e.g. 2.0 for 2-second data, 16.0 for 16-second
        data). This MUST match the actual cadence of the input arrays: it
        converts count rates to counts for the Poisson/Gaussian fit
        likelihood, so a mismatched value silently distorts every spectral
        fit (default: 16.0).
    optimization_tolerance : float
        Tolerance for spectral optimization convergence (default: 1e-3).
    max_optimization_iter : int
        Maximum iterations for spectral optimization (default: 500).

    Returns
    -------
    dict
    """
    print(f"Processing POES data from {satellite} on {date_str}", flush=True)

    # Initialize processor
    processor = POESSpectralInversion(
        dt=dt,
        optimization_tolerance=optimization_tolerance,
        max_optimization_iter=max_optimization_iter
    )
    n_time = proton_counts.shape[1]
    
    # Report processing information
    print(f"  Processing {n_time:,} time steps with {processor.interp_level} energy points", flush=True)
    
    # Report performance settings if non-default
    if n_workers > 1:
        print(f"  Using {n_workers} parallel workers", flush=True)
    if use_caching:
        print(f"  Using spectral fit caching", flush=True)
    if optimization_tolerance != 1e-3 or max_optimization_iter != 500:
        print(f"  Optimization: tol={optimization_tolerance}, max_iter={max_optimization_iter}", flush=True)
    
    # Start timing
    start_time = time.time()

    # ------------------------------------------------------------------
    # Build K matrices from response CSVs
    # Mirrors IDL:
    #   K1 = readResponse('Proton','Proton',...).response * dt
    #   K3 = readResponse('Electron','Proton',...).response * dt
    #   K4 = readResponse('Electron','Electron',...).response * dt
    #   K2 = readResponse('Proton','Electron',...).response * dt
    #   Kwfp = K1
    #   Kwfe = [[K4],[K2[*,5]]]
    # Note: readResponse() already folds in deltaE/100, so K = response * dt.
    # Python read_response_csv() returns the RAW response (no deltaE, no /100),
    # so we build K = response * deltaE[:,None] / 100 * dt  (equivalent).
    # ------------------------------------------------------------------

    proton_proton_response_path = None

    def _build_K(telescope, resp_type, n_cols_expected=None):
        """Load response CSV and return K = response * deltaE/100 * dt."""
        resp_struct = read_response_csv(
            telescope, resp_type,
            processor.min_energy, processor.max_proton_energy, processor.interp_level
        )
        resp = resp_struct['response']    # (n_energy, n_cols_csv)
        dE   = resp_struct['deltaE']      # (n_energy,)
        K = resp * dE[:, None] / 100.0 * processor.dt
        if n_cols_expected is not None and resp.shape[1] < n_cols_expected:
            raise RuntimeError(
                f"{telescope}/{resp_type} response CSV has {resp.shape[1]} columns, "
                f"expected at least {n_cols_expected}"
            )
        return K, resp_struct.get('path')

    # --- K1: Proton×Proton  (need at least P1-P6 = 6 cols) ---
    try:
        K1, proton_proton_response_path = _build_K('Proton', 'Proton', n_cols_expected=6)
        K_proton = K1[:, :6]   # (n_energy, 6) – P1-P6
    except (FileNotFoundError, RuntimeError):
        # Fallback: use per-energy GF functions
        edges = logspace(
            np.log10(processor.min_energy),
            np.log10(processor.max_proton_energy),
            processor.interp_level + 1
        )
        deltaE_fb = edges[1:] - edges[:-1]
        gf5 = get_proton_gf_array(processor.energy)   # (n_energy, 5)
        gf_p6 = gf5[:, -1:] * p6_scale
        K_proton = np.hstack([gf5, gf_p6]) * deltaE_fb[:, None] * processor.dt
        proton_proton_response_path = None

    processor.K_proton = K_proton   # used by _penalty_proton_spectrum

    # --- K3: Electron×Proton  (need at least E1-E3 = 3 cols) ---
    # This is the contamination kernel: pContam = K3^T @ p_flux  [counts]
    # protoncontamination = pContam / dt  [counts/sec]
    K3 = None
    try:
        K3_full, _ = _build_K('Electron', 'Proton', n_cols_expected=3)
        K3 = K3_full[:, :3]   # (n_energy, 3) – E1-E3 contamination
        # Override the K_ep that was set in __init__ (which was built without /100
        # double-check; the _build_K path is authoritative here).
        processor.K_ep = K3
    except (FileNotFoundError, RuntimeError):
        # K_ep already set (or None) in __init__; keep it
        K3 = processor.K_ep

    # --- K4: Electron×Electron  (E1-E3 physical channels, 3 cols) ---
    K4 = None
    try:
        K4_full, _ = _build_K('Electron', 'Electron', n_cols_expected=3)
        K4 = K4_full[:, :3]   # (n_energy, 3)
    except (FileNotFoundError, RuntimeError):
        K4 = None

    # --- K2 P6 column: Proton×Electron, column 5 (0-indexed) = P6 ---
    # IDL: Kwfe = [[K4],[K2[*,5]]]  – the E4 weighting function is the
    # P6 column of the Proton-Electron response.
    K2_p6 = None
    try:
        K2_full, _ = _build_K('Proton', 'Electron', n_cols_expected=6)
        K2_p6 = K2_full[:, 5:6]   # (n_energy, 1)
    except (FileNotFoundError, RuntimeError):
        K2_p6 = None

    # --- Kwfe = K_electron = [[K4 (E1-E3)], [K2_p6 (E4)]] ---
    if K4 is not None and K2_p6 is not None:
        K_electron = np.hstack([K4, K2_p6])   # (n_energy, 4)
    elif K4 is not None:
        # Approximate E4 GF from bowtie value (fallback)
        edges = logspace(
            np.log10(processor.min_energy),
            np.log10(processor.max_proton_energy),
            processor.interp_level + 1
        )
        deltaE_fb = edges[1:] - edges[:-1]
        gf_e4_col = np.full((K4.shape[0], 1), processor.gf_electron[3])
        K_electron = np.hstack([K4, gf_e4_col * deltaE_fb[:, None] * processor.dt])
    else:
        K_electron = None

    processor.K_electron = K_electron   # used by _penalty_electron_spectrum

    # ------------------------------------------------------------------
    # Storage arrays
    # ------------------------------------------------------------------
    proton_fluxes              = np.zeros((len(processor.energy), n_time))
    electron_fluxes            = np.zeros((len(processor.energy), n_time))
    proton_contamination       = np.zeros((3, n_time))   # E1-E3 contamination rates
    corrected_electron_counts  = np.zeros((4, n_time))   # [E1, E2, E3, E4] count rates
    e4_correction_flag         = np.zeros(n_time, dtype=np.int8)   # Flag for E4=E3 correction
    proton_model_weights       = np.zeros((4, n_time))   # [wRM, wPL, wEE, wDM]
    electron_model_weights     = np.zeros((4, n_time))

    # ------------------------------------------------------------------
    # Pre-compute Kp data for angular correction optimization
    # ------------------------------------------------------------------
    kp_data_angular = None
    kp_values_all = None
    if apply_angular_correction and electron_counts_90deg is not None and latitude is not None and time_array is not None:
        try:
            from angular_corrections import load_kp_data, get_kp_at_time, DEFAULT_RATIO_THRESHOLD
            
            # Pre-load Kp data ONCE (this was being loaded for EACH time step before!)
            kp_data_angular = load_kp_data()
            
            # Pre-compute Kp values for ALL time points in one batch
            # This eliminates the per-time-step interpolation overhead
            kp_values_all = get_kp_at_time(time_array, kp_data_angular, method='closest')
            
        except Exception as e:
            print(f"Warning: Failed to pre-load Kp data for angular correction: {e}")
            # Continue without angular correction if pre-loading fails
            kp_data_angular = None
            kp_values_all = None

    # ------------------------------------------------------------------
    # Per-time-step processing loop
    # ------------------------------------------------------------------
    for t in range(n_time):

        # ---- Step 1: Proton spectral fitting (P1-P5) ----
        # IDL: y = protoncount[0:4,i]*dt ; then amoeba on 4 models
        p_fits = processor.fit_proton_spectrum(proton_counts[0:5, t])

        # Weighted combined proton flux  (IDL: combined_flux in plot_fit_pflux)
        p_flux, p_wk = processor.get_weighted_proton_flux(p_fits)
        proton_fluxes[:, t]        = p_flux
        proton_model_weights[:, t] = p_wk

        # ---- Step 2: Predicted P1-P6 counts from proton flux ----
        # IDL: lambda = simpleForwardModel(Kwfp, combined_flux, dt, energy)
        #           = K_proton^T @ p_flux   (K_proton already has dt folded in)
        # Use IDL-equivalent matrix multiplication (K_proton already encodes deltaE and dt)
        p_lambda = K_proton.T @ p_flux   # shape (6,)

        # ---- Step 3: Proton contamination removal from E1-E3 ----
        # IDL: pContam = K3 ## protonFluxes[*,i]
        #      protoncontamination[*,i] = reform(pContam / dt)
        #      new_electronCount[*,i] = electronCount[*,i] - reform(pContam/dt)
        e_counts_corrected, p_contam = processor.remove_proton_contamination(
            p_flux, electron_counts[:, t]
        )
        proton_contamination[:, t] = p_contam   # count rates [counts/sec]

        # ---- Step 3.5: Angular correction (Selesnick) - apply AFTER proton contamination removal ---
        if apply_angular_correction and electron_counts_90deg is not None and latitude is not None and time_array is not None and kp_data_angular is not None and kp_values_all is not None:
            try:
                # Use pre-computed Kp values for this time step (MUCH faster!)
                kp_value = kp_values_all[t]
                
                # Apply angular correction to the proton-corrected electron counts
                # Sum E1-E3 channels for both telescopes
                e_counts_0deg_sum = np.sum(e_counts_corrected)  # Sum of E1-E3 proton-corrected counts
                e_counts_90deg_sum = np.sum(electron_counts_90deg[:, t])  # Sum of 90° electron counts at time t
                current_latitude = latitude[t]
                
                # Check contamination criteria (same logic as before but with pre-computed Kp)
                is_low_activity = kp_value < 3.0
                is_high_latitude = abs(current_latitude) > 50.0
                
                # Calculate ratio: 90° / 0° (avoid division by zero)
                if e_counts_0deg_sum > 0:
                    ratio = e_counts_90deg_sum / e_counts_0deg_sum
                else:
                    ratio = np.inf
                is_suspicious_ratio = ratio <= DEFAULT_RATIO_THRESHOLD
                
                # Combined mask: contaminated measurements
                contamination_mask = is_low_activity and is_high_latitude and is_suspicious_ratio
                
                # Apply linear scaling correction based on Selesnick's approach:
                # - ratio <= 1.0: Full correction (set to 0)
                # - ratio >= DEFAULT_RATIO_THRESHOLD: No correction (unchanged)
                # - 1.0 < ratio < DEFAULT_RATIO_THRESHOLD: Linear scaling between 0 and 1
                if contamination_mask and e_counts_0deg_sum > 0:
                    # Apply linear scaling: scaling_factor = clip((ratio - 1.0) / (DEFAULT_RATIO_THRESHOLD - 1.0), 0.0, 1.0)
                    if DEFAULT_RATIO_THRESHOLD <= 1.0:
                        raise ValueError(f"DEFAULT_RATIO_THRESHOLD must be > 1.0 for linear scaling. Got {DEFAULT_RATIO_THRESHOLD}")
                    scaling_factor = np.clip((ratio - 1.0) / (DEFAULT_RATIO_THRESHOLD - 1.0), 0.0, 1.0)
                    e_counts_corrected *= scaling_factor
                    
            except Exception as e:
                print(f"Warning: Failed to apply angular correction at time step {t}: {e}")
                # Continue with proton-corrected counts if angular correction fails
                pass

        # ---- Step 4: Virtual E4 = measured P6 − calculated P6 ---
        # IDL: new_electronCount = [new_electronCount,
        #                           (protonCount[5,*]-(pLambda[5,*]/dt))]
        measured_p6_rate    = proton_counts[5, t]
        calculated_p6_rate  = p_lambda[5] / processor.dt
        e4_virtual = measured_p6_rate - calculated_p6_rate
        if e4_virtual < 0.0:
            e4_virtual = 0.0

        e_counts_4ch = np.append(e_counts_corrected, e4_virtual)   # [E1,E2,E3,E4]

        # ---- Step 4.5: Check for E4 > E3 and apply correction ----
        # If E4 > E3, set E4 = E3 to prevent unphysical spectra
        if e_counts_4ch[3] > e_counts_4ch[2]:  # E4 > E3
            e_counts_4ch[3] = e_counts_4ch[2]  # Set E4 = E3
            e4_correction_flag[t] = 1       # Flag this correction
        else:
            e4_correction_flag[t] = 0       # No correction needed

        # ---- Step 5: IDL quality checks on corrected electron channels ----
        e_counts_4ch = _apply_electron_quality_checks(e_counts_4ch)
        corrected_electron_counts[:, t] = e_counts_4ch

        # ---- Step 6: Electron spectral fitting (E1-E4) ----
        e_fits = processor.fit_electron_spectrum(e_counts_4ch)
        e_flux, e_wk = processor.get_weighted_electron_flux(e_fits)
        electron_fluxes[:, t]        = e_flux
        electron_model_weights[:, t] = e_wk

        # Progress reporting - adaptive based on total time steps
        if n_time > 10000:
            report_interval = max(1000, n_time // 20)  # Report ~20 times for long runs
        elif n_time > 1000:
            report_interval = 500
        elif n_time > 100:
            report_interval = 100
        else:
            report_interval = 10
            
        if (t + 1) % report_interval == 0:
            pct = 100.0 * (t + 1) / n_time
            elapsed = time.time() - start_time
            if elapsed > 0:
                remaining = elapsed * (n_time - t - 1) / (t + 1)
                remaining_str = f"{remaining/60:.1f}m" if remaining > 60 else f"{remaining:.1f}s"
                print(f"  [{pct:.0f}%] {t+1:,}/{n_time:,} time steps | ETA: {remaining_str}", flush=True)
            else:
                print(f"  [{pct:.0f}%] {t+1:,}/{n_time:,} time steps", flush=True)

    results = {
        'energy':                     processor.energy,
        'proton_flux':                proton_fluxes,
        'electron_flux':              electron_fluxes,
        'proton_contamination':       proton_contamination,
        'corrected_electron_counts':  corrected_electron_counts,
        'e4_correction_flag':         e4_correction_flag,
        'proton_model_weights':       proton_model_weights,
        'electron_model_weights':     electron_model_weights,
        'date':                       date_str,
        'satellite':                  satellite,
        'response_paths': {
            'electron_electron': getattr(processor, 'electron_electron_response_path', None),
            'electron_proton':   getattr(processor, 'electron_proton_response_path', None),
            'proton_proton':     proton_proton_response_path,
        }
    }

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print timing summary
    if total_time > 60:
        time_str = f"{total_time/60:.1f} minutes"
    else:
        time_str = f"{total_time:.2f} seconds"
    
    time_per_step = total_time / n_time if n_time > 0 else 0
    print(f"Processing complete! Total time: {time_str} ({time_per_step*1000:.2f} ms/step)")
    return results


if __name__ == '__main__':
    # Example usage
    print("POES Spectral Flux Inversion - Python Implementation")
    print("=" * 60)
    
    # Create synthetic test data
    n_time = 100
    proton_counts = np.random.uniform(1, 100, (6, n_time))
    electron_counts = np.random.uniform(1, 100, (3, n_time))
    
    # Process the data
    results = process_poes_data(proton_counts, electron_counts, 
                               date_str='2019-09-06', satellite='m01')
    
    print("\nResults shape:")
    print(f"  Energy points: {len(results['energy'])}")
    print(f"  Proton flux shape: {results['proton_flux'].shape}")
    print(f"  Electron flux shape: {results['electron_flux'].shape}")