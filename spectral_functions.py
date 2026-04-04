"""
Spectral Functions for MPE

This module implements various spectral functions used to fit particle flux data:
- Relativistic Maxwellian
- Power Law
- Energy Exponential
- Double Relativistic Maxwellian
"""

import numpy as np
from modified_bessel import modbess_k2

# Maximum exponent argument before float64 overflows (~709.78).
# Clip to a value slightly below so exp() never produces inf.
_EXP_CLIP = 500.0

def _safe_exp(x):
    """np.exp with argument clipped to [-_EXP_CLIP, _EXP_CLIP] to avoid overflow warnings."""
    return np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP))

class SpectralFunction:
    """Base class for spectral functions"""
    
    def __call__(self, q, energy):
        """Evaluate spectral function at given energy for parameters q"""
        raise NotImplementedError
    
    def gradient(self, q, energy):
        """First derivative with respect to q"""
        raise NotImplementedError
    
    def hessian(self, q, energy):
        """Second derivative with respect to q"""
        raise NotImplementedError


class RelativisticsMaxwellianElectron(SpectralFunction):
    """Relativistic Maxwellian for electrons: f(E) = E*(1 + E/E0/2)*exp(q[0] + q[1]*E)"""
    
    def __init__(self):
        self.mass_e_keV = 510.998910
    
    def __call__(self, q, energy):
        """Evaluate relativistic Maxwellian"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        flux = energy * (1.0 + energy / self.mass_e_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        return flux
    
    def gradient(self, q, energy):
        """First derivatives of relativistic Maxwellian with respect to q"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        dfdq1 = energy * (1.0 + energy / self.mass_e_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        dfdq2 = energy * dfdq1
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item()])
        else:
            return np.array([dfdq1, dfdq2])

    def hessian(self, q, energy):
        """Second derivatives of relativistic Maxwellian with respect to q"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        d2fdq12 = energy * (1.0 + energy / self.mass_e_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        d2fdq1dq2 = energy * d2fdq12
        d2fdq22 = energy * d2fdq1dq2
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22])


class RelativisticsMaxwellianProton(SpectralFunction):
    """Relativistic Maxwellian for protons: f(E) = E*(1 + E/E0/2)*exp(q[0] + q[1]*E)"""
    
    def __init__(self):
        self.mass_p_keV = 938272.013
    
    def __call__(self, q, energy):
        """Evaluate relativistic Maxwellian"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        flux = energy * (1.0 + energy / self.mass_p_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        return flux
    
    def gradient(self, q, energy):
        """First derivatives of relativistic Maxwellian with respect to q"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        dfdq1 = energy * (1.0 + energy / self.mass_p_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        dfdq2 = energy * dfdq1
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item()])
        else:
            return np.array([dfdq1, dfdq2])

    def hessian(self, q, energy):
        """Second derivatives of relativistic Maxwellian with respect to q"""
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        d2fdq12 = energy * (1.0 + energy / self.mass_p_keV / 2.0) * _safe_exp(q[0] + q[1] * energy)
        d2fdq1dq2 = energy * d2fdq12
        d2fdq22 = energy * d2fdq1dq2
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22])


class PowerLaw(SpectralFunction):
    """Power Law: f(E) = exp(q[0] - q[1]*ln(E))"""
    
    def __call__(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        flux = _safe_exp(q[0] - q[1] * np.log(np.maximum(energy, 1e-30)))
        return flux
    
    def gradient(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        dfdq1 = _safe_exp(q[0] - q[1] * np.log(np.maximum(energy, 1e-30)))
        dfdq2 = -np.log(np.maximum(energy, 1e-30)) * dfdq1
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item()])
        else:
            return np.array([dfdq1, dfdq2])

    def hessian(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        d2fdq12 = _safe_exp(q[0] - q[1] * np.log(np.maximum(energy, 1e-30)))
        d2fdq1dq2 = -np.log(np.maximum(energy, 1e-30)) * d2fdq12
        d2fdq22 = -np.log(np.maximum(energy, 1e-30)) * d2fdq1dq2
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22])


class EnergyExponential(SpectralFunction):
    """Energy Exponential: f(E) = exp(q[0] + q[1]*E)"""
    
    def __call__(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        flux = _safe_exp(q[0] + q[1] * energy)
        return flux

    def gradient(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        dfdq1 = _safe_exp(q[0] + q[1] * energy)
        dfdq2 = energy * dfdq1
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item()])
        else:
            return np.array([dfdq1, dfdq2])

    def hessian(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        d2fdq12 = _safe_exp(q[0] + q[1] * energy)
        d2fdq1dq2 = energy * d2fdq12
        d2fdq22 = energy * d2fdq1dq2
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22])


class DoubleRelativisticsMaxwellianElectron(SpectralFunction):
    """Double Relativistic Maxwellian for electrons: f(E) = E*(1 + E/E0/2)*[exp(q[0] + q[1]*E) + exp(q[2] + q[3]*E)]"""
    
    def __init__(self):
        self.mass_e_keV = 510.998910
    
    def __call__(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        base = energy * (1.0 + energy / self.mass_e_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        flux = base * (e1 + e2)
        return flux

    def gradient(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        base = energy * (1.0 + energy / self.mass_e_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        dfdq1 = base * e1
        dfdq2 = energy * dfdq1
        dfdq3 = base * e2
        dfdq4 = energy * dfdq3
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item(), dfdq3.item(), dfdq4.item()])
        else:
            return np.array([dfdq1, dfdq2, dfdq3, dfdq4])

    def hessian(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        n = energy.size
        base = energy * (1.0 + energy / self.mass_e_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        d2fdq12 = base * e1; d2fdq1dq2 = energy * d2fdq12; d2fdq22 = energy * d2fdq1dq2
        d2fdq1dq3 = np.zeros(n); d2fdq1dq4 = np.zeros(n)
        d2fdq2dq3 = np.zeros(n); d2fdq2dq4 = np.zeros(n)
        d2fdq32 = base * e2; d2fdq3dq4 = energy * d2fdq32; d2fdq42 = energy * d2fdq3dq4
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item(), d2fdq1dq3.item(),
                           d2fdq1dq4.item(), d2fdq2dq3.item(), d2fdq2dq4.item(), d2fdq32.item(),
                           d2fdq3dq4.item(), d2fdq42.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22, d2fdq1dq3, d2fdq1dq4,
                           d2fdq2dq3, d2fdq2dq4, d2fdq32, d2fdq3dq4, d2fdq42])


class DoubleRelativisticsMaxwellianProton(SpectralFunction):
    """Double Relativistic Maxwellian for protons: f(E) = E*(1 + E/E0/2)*[exp(q[0] + q[1]*E) + exp(q[2] + q[3]*E)]"""
    
    def __init__(self):
        self.mass_p_keV = 938272.013
    
    def __call__(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        base = energy * (1.0 + energy / self.mass_p_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        flux = base * (e1 + e2)
        return flux

    def gradient(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        base = energy * (1.0 + energy / self.mass_p_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        dfdq1 = base * e1
        dfdq2 = energy * dfdq1
        dfdq3 = base * e2
        dfdq4 = energy * dfdq3
        if is_scalar:
            return np.array([dfdq1.item(), dfdq2.item(), dfdq3.item(), dfdq4.item()])
        else:
            return np.array([dfdq1, dfdq2, dfdq3, dfdq4])

    def hessian(self, q, energy):
        q = np.asarray(q, dtype=float)
        energy = np.asarray(energy, dtype=float)
        is_scalar = np.isscalar(energy) or energy.size == 1
        energy = np.atleast_1d(energy)
        n = energy.size
        base = energy * (1.0 + energy / self.mass_p_keV / 2.0)
        e1 = _safe_exp(q[0] + q[1] * energy)
        e2 = _safe_exp(q[2] + q[3] * energy)
        d2fdq12 = base * e1; d2fdq1dq2 = energy * d2fdq12; d2fdq22 = energy * d2fdq1dq2
        d2fdq1dq3 = np.zeros(n); d2fdq1dq4 = np.zeros(n)
        d2fdq2dq3 = np.zeros(n); d2fdq2dq4 = np.zeros(n)
        d2fdq32 = base * e2; d2fdq3dq4 = energy * d2fdq32; d2fdq42 = energy * d2fdq3dq4
        if is_scalar:
            return np.array([d2fdq12.item(), d2fdq1dq2.item(), d2fdq22.item(), d2fdq1dq3.item(),
                           d2fdq1dq4.item(), d2fdq2dq3.item(), d2fdq2dq4.item(), d2fdq32.item(),
                           d2fdq3dq4.item(), d2fdq42.item()])
        else:
            return np.array([d2fdq12, d2fdq1dq2, d2fdq22, d2fdq1dq3, d2fdq1dq4,
                           d2fdq2dq3, d2fdq2dq4, d2fdq32, d2fdq3dq4, d2fdq42])
