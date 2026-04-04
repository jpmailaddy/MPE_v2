"""
Penalty Functions for Spectral Fitting

This module implements the penalty/cost functions used in the spectral inversion process.
These functions are minimized to find optimal spectral parameters.
"""

import numpy as np
from spectral_functions import (RelativisticsMaxwellianElectron, RelativisticsMaxwellianProton,
                                PowerLaw, EnergyExponential,
                                DoubleRelativisticsMaxwellianElectron, DoubleRelativisticsMaxwellianProton)


class PenaltyFunctionCalculator:
    """Calculate penalty functions for spectral fitting"""
    
    def __init__(self, energy_axis, counts, count_errors, dt, kwfe=None):
        """
        Initialize penalty function calculator
        
        Parameters
        ----------
        energy_axis : array
            Energy points for spectral functions (keV)
        counts : array
            Measured count values
        count_errors : float
            Relative error in counts (e.g., 0.4 for 40%)
        dt : float
            Integration time (seconds)
        kwfe : array, optional
            Weighting function matrix (energy x channels)
        """
        self.energy = np.asarray(energy_axis, dtype=float)
        self.y = np.asarray(counts, dtype=float)
        self.dy = count_errors
        self.dt = dt
        self.kwfe = kwfe
        
        # Spectral function objects
        self.spec_functions = {
            'relmaxwell_e': RelativisticsMaxwellianElectron(),
            'relmaxwell_p': RelativisticsMaxwellianProton(),
            'powerlaw': PowerLaw(),
            'energyexponential': EnergyExponential(),
            'drelmaxwell_e': DoubleRelativisticsMaxwellianElectron(),
            'drelmaxwell_p': DoubleRelativisticsMaxwellianProton(),
        }
    
    def forward_model(self, q, spectral_function_name):
        """
        Calculate forward model: lambda = K @ f(E)
        
        Parameters
        ----------
        q : array
            Spectral parameters
        spectral_function_name : str
            Name of spectral function to use
        
        Returns
        -------
        array
            Forward model predictions for each channel
        """
        if self.kwfe is None:
            raise ValueError("Weighting function matrix (kwfe) must be provided for forward model")
        
        spec_func = self.spec_functions[spectral_function_name]
        flux = spec_func(q, self.energy)
        
        # Forward model: lambda = K @ f(E)
        lambda_pred = self.kwfe @ flux
        
        return lambda_pred
    
    def penalty_function(self, q, spectral_function_name):
        """
        Calculate penalty (cost) function
        
        Parameters
        ----------
        q : array
            Spectral parameters
        spectral_function_name : str
            Name of spectral function
        
        Returns
        -------
        float
            Penalty function value
        """
        lambda_pred = self.forward_model(q, spectral_function_name)
        
        n_channels = len(self.y)
        ell = np.zeros(n_channels)
        
        for i in range(n_channels):
            if self.y[i] < self.dy**(-2):  # Poisson statistics
                ell[i] = lambda_pred[i] - self.y[i] * np.log(lambda_pred[i])
            else:  # Gaussian statistics
                ell[i] = 0.5 * ((np.log(self.y[i]) - np.log(lambda_pred[i])) / self.dy)**2
        
        return np.sum(ell)
    
    def penalty_gradient(self, q, spectral_function_name):
        """
        Calculate gradient of penalty function with respect to q
        
        Parameters
        ----------
        q : array
            Spectral parameters
        spectral_function_name : str
            Name of spectral function
        
        Returns
        -------
        array
            Gradient vector
        """
        if self.kwfe is None:
            raise ValueError("Weighting function matrix (kwfe) must be provided")
        
        lambda_pred = self.forward_model(q, spectral_function_name)
        spec_func = self.spec_functions[spectral_function_name]
        
        # Get flux gradient: dflux/dq (n_params x n_energy)
        dfdq = spec_func.gradient(q, self.energy)
        if dfdq.ndim == 1:
            dfdq = dfdq.reshape(-1, 1)
        
        # Forward model derivative: dλ/dq = K @ (dflux/dq)
        fwd_deriv = self.kwfe @ dfdq.T  # n_channels x n_params
        
        # Penalty gradient
        n_channels = len(self.y)
        n_params = fwd_deriv.shape[1]
        dldq = np.zeros(n_params)
        
        for i in range(n_channels):
            if self.y[i] < self.dy**(-2):  # Poisson
                deriv_i = 1.0 - self.y[i] / lambda_pred[i]
            else:  # Gaussian
                deriv_i = (np.log(lambda_pred[i]) - np.log(self.y[i])) / (self.dy**2 * lambda_pred[i])
            
            dldq += deriv_i * fwd_deriv[i, :]
        
        return dldq


def first_guess_relmaxwell_electron(jflux, chan_energy):
    """
    Generate first guess for relativistic Maxwellian electron spectrum
    
    Parameters
    ----------
    jflux : array
        Initial flux guess (counts/sec)
    chan_energy : array
        Channel center energies (keV)
    
    Returns
    -------
    array
        Initial parameters [q1, q2]
    """
    mass_e_keV = 510.998910
    c_cps = 299792458.0e2
    
    # Replace zeros with small value to avoid log(0)
    jflux = np.asarray(jflux, dtype=float)
    jflux[jflux == 0] = 0.00390625
    
    # Calculate energy sum
    E_rel_sum = chan_energy * (chan_energy + 2.0 * mass_e_keV)
    
    # PSD calculation
    PSD = c_cps**2 * jflux / E_rel_sum
    
    # Linear fit in log-log space
    coeffs = np.polyfit(chan_energy, np.log(PSD), 1)
    fit_par = coeffs[::-1]  # Reverse to get [intercept, slope]
    
    q1 = np.log(2.0 * mass_e_keV * np.exp(fit_par[0]) / c_cps**2)
    Q0 = np.array([q1, fit_par[1]])
    
    return Q0


def first_guess_relmaxwell_proton(jflux, chan_energy):
    """
    Generate first guess for relativistic Maxwellian proton spectrum
    """
    mass_p_keV = 938272.013
    c_cps = 299792458.0e2
    
    jflux = np.asarray(jflux, dtype=float)
    jflux[jflux == 0] = 0.00390625
    
    E_rel_sum = chan_energy * (chan_energy + 2.0 * mass_p_keV)
    PSD = c_cps**2 * jflux / E_rel_sum
    
    coeffs = np.polyfit(chan_energy, np.log(PSD), 1)
    fit_par = coeffs[::-1]
    
    q1 = np.log(2.0 * mass_p_keV * np.exp(fit_par[0]) / c_cps**2)
    Q0 = np.array([q1, fit_par[1]])
    
    return Q0


def first_guess_powerlaw(jflux, chan_energy):
    """Generate first guess for power law spectrum"""
    jflux = np.asarray(jflux, dtype=float)
    jflux[jflux == 0] = 0.00390625
    
    coeffs = np.polyfit(np.log(chan_energy), np.log(jflux), 1)
    Q0 = np.array([coeffs[1], -coeffs[0]])  # [q1, q2]
    
    return Q0


def first_guess_exponential(jflux, chan_energy):
    """Generate first guess for energy exponential spectrum"""
    jflux = np.asarray(jflux, dtype=float)
    jflux[jflux == 0] = 0.00390625
    
    Q0 = np.polyfit(chan_energy, np.log(jflux), 1)[::-1]
    
    return Q0
