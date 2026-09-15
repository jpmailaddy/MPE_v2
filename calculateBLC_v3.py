"""
POES MPE - Bounce Loss Cone (BLC) Flux Calculation

Combines a satellite's 0-degree and 90-degree MEPED electron spectra into a
Bounce Loss Cone (BLC) differential flux: at each energy/time, a y = A*sin(pitch)
curve is fit through the two detectors' flux measurements, and that fit is
evaluated at the BLC boundary angle and integrated over the BLC solid angle
(assuming a sin(pitch) angular distribution, i.e. n=1 in Evans & Greer's
equation 2.3.3) to get an omnidirectional flux within the loss cone.

Ported from the IDL implementation in IDL_Code/calculateBLC_v3.pro
(calculateBLC_v1 procedure and fitSine function, by Joshua Pettit). See the
docstrings below for the handful of places this port deviates from -- or had
to make an explicit choice about -- IDL quirks/bugs in the original.
"""

import numpy as np
import netCDF4 as nc


def _reflect_pitch(pitch):
    """Reflect pitch angles > 90 deg back into [0, 90] by mirror symmetry of sin().

    Matches IDL: ``if p gt 90. then p = 90.-(p-90.)``.
    """
    pitch = np.asarray(pitch, dtype=float)
    return np.where(pitch > 90.0, 90.0 - (pitch - 90.0), pitch)


def fit_sine_amplitude(m0, m90, p0, p90):
    """
    Fit y = A*sin(pitch) through the 0-degree and 90-degree detector measurements.

    Ports IDL's ``fitSine()``. The IDL version calls ``CURVEFIT`` on a
    single-parameter model that is linear in A, which (with an exact analytic
    derivative and no weighting -- the ``w`` argument is passed undefined/unset
    in the active code path) converges to the exact unweighted least-squares
    solution in one step. That closed form is computed directly here instead
    of reimplementing iterative nonlinear least squares:

        A = (m0*sin(p0) + m90*sin(p90)) / (sin(p0)**2 + sin(p90)**2)

    IDL only takes this two-point fit when the two measurements have the
    "physically expected" ordering (higher pitch angle -> higher flux, in
    either direction). In every other case (equal pitches, equal fluxes, or
    an inverted ordering), the original code falls back to a single-point
    amplitude derived from the 0-degree detector alone: ``A = m0/sin(p0)``.
    That fallback is reproduced here via ``use_two_point`` below.

    One case is never handled by the original IDL: ``p0 == p90`` and
    ``m0 == m90`` simultaneously satisfies none of ``fitSine``'s branch
    conditions, so IDL's local ``A`` is left undefined at the `return, A`
    statement. This port treats it the same as IDL's other degenerate
    branches (``A = m0/sin(p0)``), which correctly resolves to 0 when both
    detectors read zero flux -- the common case in practice.

    Parameters
    ----------
    m0, m90 : float or ndarray
        0-degree / 90-degree electron differential flux.
    p0, p90 : float or ndarray
        0-degree / 90-degree pitch angle (degrees).

    Returns
    -------
    float or ndarray
        Fitted amplitude A (same broadcast shape as the inputs).
    """
    m0 = np.clip(np.asarray(m0, dtype=float), 0.0, None)
    m90 = np.clip(np.asarray(m90, dtype=float), 0.0, None)
    p0 = _reflect_pitch(p0)
    p90 = _reflect_pitch(p90)

    sin_p0 = np.sin(np.deg2rad(p0))
    sin_p90 = np.sin(np.deg2rad(p90))

    with np.errstate(invalid='ignore', divide='ignore'):
        a_two_point = (m0 * sin_p0 + m90 * sin_p90) / (sin_p0**2 + sin_p90**2)
        a_single_point = m0 / sin_p0

    use_two_point = ((p90 > p0) & (m90 > m0)) | ((p0 > p90) & (m0 > m90))
    a = np.where(use_two_point, a_two_point, a_single_point)

    return a if a.ndim > 0 else float(a)


def calculate_blc_flux(flux_0deg, flux_90deg, pitch_0deg, pitch_90deg, blc_angle_deg):
    """
    Compute BLC differential flux over a full (energy, time) grid.

    Ports the per-energy/per-time loop body of IDL's ``calculateBLC_v1``
    (the ``blc_flux_v5`` branch -- the only one actually written to the
    original output file; an earlier, unused ``blc_flux`` formula existed
    in the IDL and was not carried over here).

    Parameters
    ----------
    flux_0deg, flux_90deg : ndarray, shape (n_energy, n_time)
        Fitted electron differential flux for the 0-deg / 90-deg telescopes
        (each telescope's 'Ecounts' spectral-inversion output). Any
        already-known-bad spectra (e.g. from the raw-count quality gate in
        :func:`calculate_blc_for_day`) should be zeroed before calling this.
    pitch_0deg, pitch_90deg : ndarray, shape (n_time,)
        Pitch angle (degrees) for each telescope.
    blc_angle_deg : ndarray, shape (n_time,)
        Bounce loss cone boundary angle (degrees).

    Returns
    -------
    ndarray, shape (n_energy, n_time)
        BLC differential flux (units: flux units of the input, integrated
        over the BLC solid angle -- e.g. counts/cm^2/s/keV if the input is
        counts/cm^2/s/sr/keV).
    """
    m0 = np.array(flux_0deg, dtype=float, copy=True)
    m90 = np.array(flux_90deg, dtype=float, copy=True)
    m0[~np.isfinite(m0)] = 0.0
    m90[~np.isfinite(m90)] = 0.0

    p0 = np.broadcast_to(np.asarray(pitch_0deg, dtype=float), m0.shape).copy()
    p90 = np.broadcast_to(np.asarray(pitch_90deg, dtype=float), m0.shape).copy()

    # IDL nudges tm90/tp90 by +0.001 whenever tm00 == tm90 exactly, to steer
    # fitSine away from its degenerate equal-flux branch.
    tie = (m0 == m90)
    m90 = np.where(tie, m90 + 0.001, m90)
    p90 = np.where(tie, p90 + 0.001, p90)

    # IDL's "Huge Numbers" guard: zero tm00, tm90, tp00 when either flux
    # exceeds 1e10. NOTE: the original sets `tm90 = 0.` a second time where
    # `tp90 = 0.` was very likely intended (a copy/paste typo) -- tp90 is
    # left untouched by the original code, so it's left untouched here too
    # for fidelity. Worth revisiting in the "additional improvements" pass.
    huge = (m0 > 1e10) | (m90 > 1e10)
    m0 = np.where(huge, 0.0, m0)
    m90 = np.where(huge, 0.0, m90)
    p0 = np.where(huge, 0.0, p0)

    # IDL only computes the fit where tp90 is finite; q stays at its
    # zero-initialized default otherwise.
    valid = np.isfinite(p90)
    a = np.zeros_like(m0)
    a[valid] = fit_sine_amplitude(m0[valid], m90[valid], p0[valid], p90[valid])

    alpha_rad = np.deg2rad(np.broadcast_to(np.asarray(blc_angle_deg, dtype=float), m0.shape))
    return 2.0 * np.pi * a * np.sin(alpha_rad) ** 3 / 3.0


def _read_var(ds, name):
    """Read a NetCDF variable if present, else return None."""
    if name in ds.variables:
        return np.asarray(ds.variables[name][:])
    return None


def calculate_blc_for_day(file_0deg, file_90deg, output_file=None):
    """
    Combine a 0-degree and 90-degree telescope spectral-inversion output file
    into a single Bounce Loss Cone (BLC) differential electron flux product.

    Ports the per-day file-processing body of IDL's ``calculateBLC_v1``.

    Parameters
    ----------
    file_0deg, file_90deg : str
        Paths to the 0-degree / 90-degree telescope NetCDF output files
        (as written by ``run_poes_processing_v2.py`` / ``data_io.py``), for
        the same satellite and date.
    output_file : str, optional
        If given, write the BLC product to this NetCDF file
        (see :func:`write_blc_netcdf`).

    Returns
    -------
    dict
        Keys: 'energy', 'time', 'rtime', 'geogLat', 'geogLon', 'foflLat',
        'foflLon', 'MLT', 'lValue', 'BLC_Angle', 'BLC_Flux'.

    Notes
    -----
    The original IDL pipeline (``spectral_fits_00.pro``) also wrote an
    'Eerror' (flux uncertainty) and a 'flag' (suspicious-spectrum) variable
    per telescope, and ``calculateBLC_v3.pro`` reads both. Neither is
    currently produced by this repo's Python spectral-inversion output
    (``data_io.py`` / ``spectral_inversion_main.py``). IDL treats a missing
    'flag' variable as "no BLC data this day" and skips the whole file --
    replicating that here would make this function skip every day against
    current output, so instead: missing 'flag' degrades to an all-clear
    (``Flag = 0``) rather than skipping the day, and 'Eerror' currently
    isn't propagated into the output at all (the IDL wrote it straight
    through as 'BLC_Flux_Error', but that variable was already commented out
    of the IDL's own NCDF_VARDEF/NCDF_VARPUT calls, so nothing is lost by
    leaving it out here). If 'Eerror' output is wanted later, it needs to be
    added to the upstream spectral-inversion writer first.
    """
    with nc.Dataset(file_0deg, 'r') as ds0, nc.Dataset(file_90deg, 'r') as ds90:
        time0 = np.asarray(ds0.variables['time'][:])
        time90 = np.asarray(ds90.variables['time'][:])

        n0, n90 = len(time0), len(time90)
        if n0 < n90:
            raise ValueError(
                f"{file_90deg} has more time steps ({n90}) than {file_0deg} ({n0}) -- "
                "the original IDL treats this as unrecoverable data corruption."
            )
        n_use = n90  # matches IDL: 00-degree arrays are trimmed to the 90-degree length

        def trim(name, ds=ds0):
            arr = _read_var(ds, name)
            return None if arr is None else arr[..., :n_use]

        ecounts0 = trim('Ecounts')
        ecounts90 = trim('Ecounts', ds90)
        eocounts0 = trim('EOcounts')
        eocounts90 = trim('EOcounts', ds90)
        pitch0 = trim('pitch')
        pitch90 = trim('pitch', ds90)
        blc_angle = trim('BLC_Angle')
        energy = np.asarray(ds0.variables['energy'][:])
        rtime = trim('rtime')
        geog_lat = trim('geogLat')
        geog_lon = trim('geogLon')
        fofl_lat = trim('foflLat')
        fofl_lon = trim('foflLon')
        mlt = trim('MLT')
        l_value = trim('lValue')
        time_out = time90  # IDL writes the 90-degree file's time array (time2)


    # Quality gate: zero the whole fitted spectrum at a time step if that
    # telescope's raw E1 count is exactly zero (matches IDL's check on
    # m00_original[0,i] / m90_original[0,i]).
    ecounts0 = ecounts0.copy()
    ecounts90 = ecounts90.copy()
    ecounts0[:, eocounts0[0, :] == 0.0] = 0.0
    ecounts90[:, eocounts90[0, :] == 0.0] = 0.0

    blc_flux = calculate_blc_flux(ecounts0, ecounts90, pitch0, pitch90, blc_angle)

    result = {
        'energy': energy,
        'time': time_out,
        'rtime': rtime,
        'geogLat': geog_lat,
        'geogLon': geog_lon,
        'foflLat': fofl_lat,
        'foflLon': fofl_lon,
        'MLT': mlt,
        'lValue': l_value,
        'BLC_Angle': blc_angle,
        'BLC_Flux': blc_flux,
    }

    if output_file is not None:
        write_blc_netcdf(output_file, result)

    return result


def write_blc_netcdf(output_file, result):
    """
    Write a BLC product dict (as returned by :func:`calculate_blc_for_day`) to
    a NetCDF file, matching the variable/attribute schema of IDL's output.
    """
    with nc.Dataset(output_file, 'w') as ds:
        ds.createDimension('energy', len(result['energy']))
        ds.createDimension('time', None)

        def var(name, dims, dtype, data, full_name, units):
            v = ds.createVariable(name, dtype, dims)
            v.Full_Name = full_name
            v.Units = units
            v[:] = data
            return v

        var('time', ('time',), 'f8', result['time'], 'Time of measurement', 'milliseconds since 1970')
        var('rtime', ('time',), 'f8', result['rtime'], 'Time of measurement', 'hours')
        var('geogLat', ('time',), 'f8', result['geogLat'], 'Satellite Geographic Latitude', 'degrees -90 to 90')
        var('geogLon', ('time',), 'f8', result['geogLon'], 'Satellite Geographic Longitude', 'degrees 0 to 360')
        var('foflLat', ('time',), 'f8', result['foflLat'], 'Satellite Foot of the Field Line Geographic Latitude', 'degrees -90 to 90')
        var('foflLon', ('time',), 'f8', result['foflLon'], 'Satellite Foot of the Field Line Geographic Longitude', 'degrees 0 to 360')
        var('MLT', ('time',), 'f8', result['MLT'], 'Satellite Magnetic Local Time', 'hours')
        var('lValue', ('time',), 'f8', result['lValue'], 'Satellite L-Value', ' ')
        var('energy', ('energy',), 'f4', result['energy'], 'Center Energy of Electron Fluxes', 'KeV')
        var('BLC_Flux', ('energy', 'time'), 'f8', result['BLC_Flux'], 'Bounce Loss Cone Differential Flux', 'counts/cm2/s/keV')
        var('BLC_Angle', ('time',), 'f8', result['BLC_Angle'], 'Calculated Bounce Loss Cone Angle', 'degrees')
