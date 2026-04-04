"""
Geometric Factors and Response Functions for POES Instruments

This module contains lookup tables and interpolation functions for:
- Electron geometric factors (MEPED)
- Proton geometric factors (MEPED)
- Response functions (G-factors)
"""

import numpy as np
from scipy.interpolate import interp1d

# Notes:
# - The gf_me*_recommended and gf_mp*_recommended functions provide the
#   geometric factor (GF) as a function of energy for a single channel. The
#   arrays inside those functions are energy breakpoints and GF values for
#   interpolation, not channel counts. For example, gf_me1_recommended(energy)
#   returns the GF for electron channel E1 evaluated at `energy`.
# - The code defines GF functions for 5 electron channels (E1..E5) and 5
#   proton channels (P1..P5). The P6 proton channel is usually handled
#   separately (e.g., via a response matrix or approximated from P5) in the
#   processing pipeline; that's why the proton GF array functions return 5
#   columns (P1..P5) rather than 6.


class GeometricFactorsElectron:
    """Geometric factors for electron telescopes (MEPED)"""
    
    @staticmethod
    def gf_me1_recommended(energy):
        """GF for MEPED Electron Channel 1 (E1)"""
        xenergy = np.array([20.0, 25.0, 35.0, 45.0, 60.0, 200.0, 500.0, 2000.0])
        gf = np.array([8.2e-8, 3.0e-5, 1.0e-2, 1.0e-2, 1.0e-3, 1.0e-3, 2.0e-4, 2.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_me2_recommended(energy):
        """GF for MEPED Electron Channel 2 (E2)"""
        xenergy = np.array([40.0, 45.0, 55.0, 95.0, 110.0, 250.0, 500.0, 2000.0])
        gf = np.array([1.03e-6, 3.0e-5, 1.0e-2, 1.0e-2, 1.0e-3, 1.0e-3, 4.0e-4, 4.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_me3_recommended(energy):
        """GF for MEPED Electron Channel 3 (E3)"""
        xenergy = np.array([55.0, 60.0, 75.0, 90.0, 95.0, 105.0, 200.0, 500.0, 2000.0])
        gf = np.array([5.75e-7, 1.42e-6, 2.24e-6, 1.39e-6, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-3, 1.0e-3])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_me4_recommended(energy):
        """GF for MEPED Electron Channel 4 (E4)"""
        xenergy = np.array([114.0, 190.0, 220.0, 350.0, 500.0, 800.0, 2000.0])
        gf = np.array([4.87e-4, 1.0e-3, 1.0e-2, 1.0e-2, 3.0e-3, 6.0e-3, 1.0e-2])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_me5_recommended(energy):
        """GF for MEPED Electron Channel 5 (E5)"""
        xenergy = np.array([114.0, 168.0, 261.0, 350.0, 400.0, 600.0, 1000.0, 2000.0])
        gf = np.array([1.0e-7, 6.94e-7, 2.99e-7, 1.0e-3, 1.0e-2, 1.0e-2, 4.0e-3, 3.0e-3])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))


class GeometricFactorsProton:
    """Geometric factors for proton telescopes (MEPED)"""
    
    @staticmethod
    def gf_mp1_recommended(energy):
        """GF for MEPED Proton Channel 1 (P1)"""
        xenergy = np.array([75., 85., 105., 115.])
        gf = np.array([1.0e-4, 1.0e-2, 1.0e-2, 1.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_mp2_recommended(energy):
        """GF for MEPED Proton Channel 2 (P2)"""
        xenergy = np.array([105., 115., 165., 175.])
        gf = np.array([1.0e-4, 1.0e-2, 1.0e-2, 1.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_mp3_recommended(energy):
        """GF for MEPED Proton Channel 3 (P3)"""
        xenergy = np.array([165., 175., 245., 255.])
        gf = np.array([1.0e-4, 1.0e-2, 1.0e-2, 1.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_mp4_recommended(energy):
        """GF for MEPED Proton Channel 4 (P4)"""
        xenergy = np.array([245., 255., 345., 355.])
        gf = np.array([1.0e-4, 1.0e-2, 1.0e-2, 1.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))
    
    @staticmethod
    def gf_mp5_recommended(energy):
        """GF for MEPED Proton Channel 5 (P5)"""
        xenergy = np.array([345., 355., 795., 805.])
        gf = np.array([1.0e-4, 1.0e-2, 1.0e-2, 1.0e-4])
        
        if energy < xenergy[0] or energy > xenergy[-1]:
            return 0.0
        
        return 10.0**np.interp(np.log10(energy), np.log10(xenergy), np.log10(gf))


def get_electron_gf_array(energy_array, n_channels=5):
    """Get geometric factor array for electron channels.

    Parameters
    ----------
    energy_array : array-like
        Energy grid (keV) where GF will be evaluated.
    n_channels : int, optional
        Number of electron channels to return (1-based). Default 5.

    Returns
    -------
    gf : ndarray
        Array shape (n_energy, n_channels) with geometric factors.

    Notes
    -----
    Historically MEPED instruments have five electron GF definitions available
    (E1..E5). The processing pipeline commonly uses E1..E4 (E4 is the virtual
    channel derived from P6), so callers may request n_channels=4 to match
    pipeline usage. E5 is provided for completeness for other use-cases.
    """
    # default to 5 channels for backward compatibility
    def _inner(energy_array, n_channels=5):
        gf_electrons = GeometricFactorsElectron()
        gf_funcs = [gf_electrons.gf_me1_recommended, gf_electrons.gf_me2_recommended,
                    gf_electrons.gf_me3_recommended, gf_electrons.gf_me4_recommended,
                    gf_electrons.gf_me5_recommended]

        if n_channels < 1 or n_channels > len(gf_funcs):
            raise ValueError(f'n_channels must be between 1 and {len(gf_funcs)}')

        gf = np.zeros((len(energy_array), n_channels))
        for i, func in enumerate(gf_funcs[:n_channels]):
            for j, e in enumerate(energy_array):
                gf[j, i] = func(e)

        return gf

    # Keep the simple callable API for backward compatibility (default n_channels=5)
    return _inner(energy_array, n_channels)


def get_proton_gf_array(energy_array):
    """Get geometric factor array for 6 proton channels"""
    gf_protons = GeometricFactorsProton()
    gf_funcs = [gf_protons.gf_mp1_recommended, gf_protons.gf_mp2_recommended,
                gf_protons.gf_mp3_recommended, gf_protons.gf_mp4_recommended,
                gf_protons.gf_mp5_recommended]
    
    gf = np.zeros((len(energy_array), 5))
    for i, func in enumerate(gf_funcs):
        for j, e in enumerate(energy_array):
            gf[j, i] = func(e)
    
    return gf


def logspace(a, b, n):
    """Create n logarithmically-spaced values between 10^a and 10^b.

    Equivalent to IDL ``logspace(A, B, N)``:
        ``DINDGEN(N) / (N-1) * (B-A) + A`` mapped through ``10^``.
    Produces n edge points from 10^a to 10^b (inclusive).
    """
    return 10.0 ** np.linspace(a, b, n)


def logspace_midpoints(a, b, n):
    """Return n geometric-mean midpoints of n+1 log-spaced edges between 10^a and 10^b.

        edges     = logspace(alog10(minE), alog10(maxE), interp+1)   ; n+1 edges
        midpoints[i] = sqrt(edges[i] * edges[i+1])                   ; n midpoints

    Parameters
    ----------
    a : float
        log10 of the minimum energy (e.g. ``np.log10(25.)``).
    b : float
        log10 of the maximum energy (e.g. ``np.log10(10000.)``).
    n : int
        Number of desired midpoints (== ``interp_level``).

    Returns
    -------
    midpoints : ndarray, shape (n,)
        Geometric-mean midpoints of the n+1 log-spaced edges.
    """
    edges = logspace(a, b, n + 1)
    return np.sqrt(edges[:-1] * edges[1:])