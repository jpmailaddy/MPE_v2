"""
Data I/O Functions for MPE

This module handles reading and writing data in various formats:
- NetCDF files (output format)
- CDF files (legacy input format)
"""

import numpy as np
import netCDF4 as nc
from datetime import datetime


def calculate_rtime(time_ms, year=None, day=None):
    """
    Calculate fractional hours since start of day (UTC) — vectorized.

    Parameters
    ----------
    time_ms : array-like
        Time in milliseconds since 1970-01-01 00:00:00 UTC (Unix epoch × 1000).
    year, day : ignored (kept for API compatibility)

    Returns
    -------
    rtime : ndarray
        Fractional hours since midnight UTC, range [0, 24).
    """
    time_ms = np.asarray(time_ms, dtype=np.float64)
    # Seconds since epoch
    time_sec = time_ms / 1000.0
    # Seconds within each UTC day  (mod 86400)
    seconds_since_midnight = np.mod(time_sec, 86400.0)
    return seconds_since_midnight / 3600.0


class NetCDFWriter:
    """Write processed POES flux data to NetCDF files"""
    
    def __init__(self, filename):
        """
        Initialize NetCDF writer
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        self.filename = filename
        self.ds = None
    
    def create_file(self, energy, n_time_steps, n_proton_channels=6, n_electron_channels=4):
        """
        Create NetCDF file structure
        
        Parameters
        ----------
        energy : array
            Energy grid (keV)
        n_time_steps : int
            Number of time points
        n_proton_channels : int
            Number of proton channels
        n_electron_channels : int
            Number of electron channels
        """
        self.ds = nc.Dataset(self.filename, 'w', format='NETCDF4')
        
        # Create dimensions
        self.ds.createDimension('energy', len(energy))
        self.ds.createDimension('proton_telescopes', n_proton_channels)
        self.ds.createDimension('electron_telescopes', 3)
        self.ds.createDimension('electron_telescopes_and_E4', n_electron_channels)
        self.ds.createDimension('2_Parameter_Spectrum', 2)
        self.ds.createDimension('4_Parameter_Spectrum', 4)
        self.ds.createDimension('number_of_spectra', 4)
        self.ds.createDimension('time', None)  # Unlimited time dimension
        
        # Create energy variable
        energy_var = self.ds.createVariable('energy', 'f4', ('energy',))
        energy_var.setncattr('Full_Name', 'Center Energy of Electron/Proton Fluxes')
        energy_var.setncattr('Units', 'keV')
        energy_var[:] = energy
        
        # Create time variable
        time_var = self.ds.createVariable('time', 'f8', ('time',))
        time_var.setncattr('Full_Name', 'Time of measurement')
        time_var.setncattr('Units', 'milliseconds since 1970-01-01 00:00:00 UTC')
        time_var.setncattr('standard_name', 'time')
        time_var.setncattr('calendar', 'gregorian')

        # Create rtime variable (fractional hours since start of day)
        rtime_var = self.ds.createVariable('rtime', 'f8', ('time',))
        rtime_var.setncattr('Full_Name', 'Fractional hours since start of day')
        rtime_var.setncattr('Units', 'hours')
        rtime_var.setncattr('Description', 'Useful for day-to-day checking; ranges from 0 to 24')
        
        # Create geolocation variables
        geog_lat = self.ds.createVariable('geogLat', 'f8', ('time',))
        geog_lat.setncattr('Full_Name', 'Satellite Geographic Latitude')
        geog_lat.setncattr('Units', 'degrees -90 to 90')
        
        geog_lon = self.ds.createVariable('geogLon', 'f8', ('time',))
        geog_lon.setncattr('Full_Name', 'Satellite Geographic Longitude')
        geog_lon.setncattr('Units', 'degrees 0 to 360')
        
        fofl_lat = self.ds.createVariable('foflLat', 'f8', ('time',))
        fofl_lat.setncattr('Full_Name', 'Satellite Foot of the Field Line Geographic Latitude')
        fofl_lat.setncattr('Units', 'degrees -90 to 90')
        
        fofl_lon = self.ds.createVariable('foflLon', 'f8', ('time',))
        fofl_lon.setncattr('Full_Name', 'Satellite Foot of the Field Line Geographic Longitude')
        fofl_lon.setncattr('Units', 'degrees 0 to 360')
        
        # Magnetic parameters
        mlt = self.ds.createVariable('MLT', 'f8', ('time',))
        mlt.setncattr('Full_Name', 'Satellite Magnetic Local Time')
        mlt.setncattr('Units', 'degrees 0 to 360')
        
        lvalue = self.ds.createVariable('lValue', 'f8', ('time',))
        lvalue.setncattr('Full_Name', 'Satellite L-Value')
        lvalue.setncattr('Units', ' ')
        
        pitch = self.ds.createVariable('pitch', 'f8', ('time',))
        pitch.setncattr('Full_Name', 'Pitch Angle')
        pitch.setncattr('Units', 'degrees -180 to 180')
        
        # Magnetic field at satellite and foot
        bfofl = self.ds.createVariable('Bfofl', 'f8', ('time',))
        bfofl.setncattr('Full_Name', 'Magnetic Field at Foot of Field Line')
        bfofl.setncattr('Units', 'Tesla')
        
        blocal = self.ds.createVariable('Blocal', 'f8', ('time',))
        blocal.setncattr('Full_Name', 'Magnetic Field at Satellite')
        blocal.setncattr('Units', 'Tesla')
        
        blc_angle = self.ds.createVariable('BLC_Angle', 'f8', ('time',))
        blc_angle.setncattr('Full_Name', 'Bounce Loss Cone Angle')
        blc_angle.setncattr('Units', 'degrees 0 to 90')
        
        # Flux data
        ecounts = self.ds.createVariable('Ecounts', 'f8', ('energy', 'time'))
        ecounts.setncattr('Full_Name', 'Corrected Electron Flux Rate')
        ecounts.setncattr('Units', 'N/cm^2/sr/keV')
        ecounts.setncattr('Bad_Data_Value', -999.0)
        
        pcounts = self.ds.createVariable('Pcounts', 'f8', ('energy', 'time'))
        pcounts.setncattr('Full_Name', 'Corrected Proton Flux Rate')
        pcounts.setncattr('Units', 'N/cm^2/sr/keV')
        pcounts.setncattr('Bad_Data_Value', -999.0)
        
        # Count rates
        eocounts = self.ds.createVariable('EOcounts', 'f8', ('electron_telescopes', 'time'))
        eocounts.setncattr('Full_Name', 'Original Electron Count Rate')
        eocounts.setncattr('Units', 'counts/sec')
        
        pocounts = self.ds.createVariable('POcounts', 'f8', ('proton_telescopes', 'time'))
        pocounts.setncattr('Full_Name', 'Original Proton Count Rate')
        pocounts.setncattr('Units', 'counts/sec')
        
        # Proton contamination estimated for electron channels (E1-E3)
        pcont = self.ds.createVariable('ProtonContamination', 'f8', ('electron_telescopes', 'time'))
        pcont.setncattr('Full_Name', 'Estimated Proton Contamination in Electron Channels')
        pcont.setncattr('Units', 'counts/sec')
        pcont.setncattr('Description', 'Estimated proton counts (to be subtracted) for electron channels E1-E3')
        pcont.setncattr('Bad_Data_Value', -999.0)

        # Corrected electron counts including virtual E4 (E1-E4)
        corrected_eocounts = self.ds.createVariable('CorrectedEOcounts', 'f8', ('electron_telescopes_and_E4', 'time'))
        corrected_eocounts.setncattr('Full_Name', 'Corrected Electron Count Rate (including E4)')
        corrected_eocounts.setncattr('Units', 'counts/sec')
        corrected_eocounts.setncattr('Description', 'Electron count rates after proton contamination removal, includes virtual E4 derived from P6')
        corrected_eocounts.setncattr('Bad_Data_Value', -999.0)
        
        # Spectral parameters
        ermq = self.ds.createVariable('ERMq', 'f8', ('2_Parameter_Spectrum', 'time'))
        ermq.setncattr('Full_Name', 'Parameters for Electron Relativistic Maxwellian Spectrum')
        ermq.setncattr('Spectrum_Equation', 'f(E) = E*(1 + E/E0/2)*exp(q1 + q2*E)')
        ermq.setncattr('Rest_Energy', 'E0 = 511 keV')
        
        eplq = self.ds.createVariable('EPLq', 'f8', ('2_Parameter_Spectrum', 'time'))
        eplq.setncattr('Full_Name', 'Parameters for Electron Power Law Spectrum')
        eplq.setncattr('Spectrum_Equation', 'f(E) = exp(q1 - q2*ln(E))')
        
        eeeq = self.ds.createVariable('EEEq', 'f8', ('2_Parameter_Spectrum', 'time'))
        eeeq.setncattr('Full_Name', 'Parameters for Electron Energy Exponential Spectrum')
        eeeq.setncattr('Spectrum_Equation', 'f(E) = exp(q1 + q2*E)')
        
        edmq = self.ds.createVariable('EDMq', 'f8', ('4_Parameter_Spectrum', 'time'))
        edmq.setncattr('Full_Name', 'Parameters for Electron Double Relativistic Maxwellian Spectrum')
        edmq.setncattr('Spectrum_Equation', 'f(E) = E*(1 + E/E0/2)*[exp(q1 + q2*E)+exp(q3 + q4*E)]')
        edmq.setncattr('Rest_Energy', 'E0 = 511 keV')
        
        # Global attributes
        self.ds.setncattr('Author', 'Joshua Pettit, NASA GSFC/GMU MPEv2 (Python Port)')
        self.ds.setncattr('Date_Created', datetime.now().isoformat())
    
    def write_data(self, time_idx, data_dict):
        """
        Write data to a time index

        Parameters
        ----------
        time_idx : int
            Time index to write to
        data_dict : dict
            Dictionary containing data arrays
        """
        if self.ds is None:
            raise ValueError("File not created. Call create_file first.")

        var_map = {
            'time': 'time', 'rtime': 'rtime',
            'geog_lat': 'geogLat', 'geog_lon': 'geogLon',
            'fofl_lat': 'foflLat', 'fofl_lon': 'foflLon',
            'mlt': 'MLT', 'lvalue': 'lValue', 'pitch': 'pitch',
            'bfofl': 'Bfofl', 'blocal': 'Blocal', 'blc_angle': 'BLC_Angle',
            'electron_flux': 'Ecounts', 'proton_flux': 'Pcounts',
            'electron_counts': 'EOcounts', 'proton_counts': 'POcounts',
            'proton_contamination': 'ProtonContamination',
            'corrected_electron_counts': 'CorrectedEOcounts',
            'e_rm_q': 'ERMq', 'e_pl_q': 'EPLq',
            'e_ee_q': 'EEEq', 'e_dm_q': 'EDMq',
        }

        for key, var_name in var_map.items():
            if key in data_dict and var_name in self.ds.variables:
                var = self.ds.variables[var_name]
                val = data_dict[key]
                if val is None:
                    continue
                if key in ['electron_flux', 'proton_flux', 'electron_counts', 'proton_counts',
                           'e_rm_q', 'e_pl_q', 'e_ee_q', 'e_dm_q',
                           'proton_contamination', 'corrected_electron_counts']:
                    var[:, time_idx] = val
                else:
                    var[time_idx] = val

    def write_data_batch(self, data_dict):
        """Write all time steps at once (much faster than calling write_data per step).

        Parameters
        ----------
        data_dict : dict
            Same keys as ``write_data`` but each value is the full array over time.
            1-D time variables must have shape (n_time,).
            2-D channel×time variables must have shape (n_channels, n_time).
        """
        if self.ds is None:
            raise ValueError("File not created. Call create_file first.")

        scalar_vars = {
            'time': 'time', 'rtime': 'rtime',
            'geog_lat': 'geogLat', 'geog_lon': 'geogLon',
            'fofl_lat': 'foflLat', 'fofl_lon': 'foflLon',
            'mlt': 'MLT', 'lvalue': 'lValue', 'pitch': 'pitch',
            'bfofl': 'Bfofl', 'blocal': 'Blocal', 'blc_angle': 'BLC_Angle',
        }
        array_vars = {
            'electron_flux': 'Ecounts', 'proton_flux': 'Pcounts',
            'electron_counts': 'EOcounts', 'proton_counts': 'POcounts',
            'proton_contamination': 'ProtonContamination',
            'corrected_electron_counts': 'CorrectedEOcounts',
            'e_rm_q': 'ERMq', 'e_pl_q': 'EPLq',
            'e_ee_q': 'EEEq', 'e_dm_q': 'EDMq',
        }
        for key, var_name in scalar_vars.items():
            val = data_dict.get(key)
            if val is None or var_name not in self.ds.variables:
                continue
            self.ds.variables[var_name][:] = np.asarray(val)

        for key, var_name in array_vars.items():
            val = data_dict.get(key)
            if val is None or var_name not in self.ds.variables:
                continue
            # Stored as (channels, time) in the NetCDF, but passed in same shape
            self.ds.variables[var_name][:] = np.asarray(val)

    def close(self):
        """Close the NetCDF file"""
        if self.ds is not None:
            self.ds.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NetCDFReader:
    """Read POES NetCDF flux data (output format)"""
    
    def __init__(self, filename):
        """
        Initialize NetCDF reader
        
        Parameters
        ----------
        filename : str
            Input filename
        """
        self.filename = filename
        self.ds = nc.Dataset(filename, 'r')
    
    def read_energy(self):
        """Read energy grid"""
        return self.ds.variables['energy'][:]
    
    def read_time(self):
        """Read time array"""
        return self.ds.variables['time'][:]
    
    def read_electron_flux(self):
        """Read electron flux"""
        return self.ds.variables['Ecounts'][:, :]
    
    def read_proton_flux(self):
        """Read proton flux"""
        return self.ds.variables['Pcounts'][:, :]
    
    def close(self):
        """Close the dataset"""
        self.ds.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class POESRawDataReader:
    """Read raw POES particle count data from dual NetCDF input files
    
    The IDL code reads from two separate files:
    1. *_raw.nc - Contains raw particle count channels (E1-E3, P1-P6)
    2. *_proc.nc - Contains processed geolocation and magnetic field data
    """
    
    def __init__(self, raw_file, proc_file):
        """
        Initialize POES raw data reader
        
        Parameters
        ----------
        raw_file : str
            Path to *_raw.nc file with particle counts
        proc_file : str
            Path to *_proc.nc file with geolocation/magnetic data
        """
        self.raw_file = raw_file
        self.proc_file = proc_file
        self.ds_raw = nc.Dataset(raw_file, 'r')
        self.ds_proc = nc.Dataset(proc_file, 'r')
    
    def read_electron_counts(self):
        """
        Read electron count channels (E1, E2, E3)
        
        Returns
        -------
        electron_counts : ndarray, shape (3, n_time)
            Electron counts for channels E1, E2, E3
        """
        e1 = self.ds_raw.variables['mep_ele_tel0_cps_e1'][:]
        e2 = self.ds_raw.variables['mep_ele_tel0_cps_e2'][:]
        e3 = self.ds_raw.variables['mep_ele_tel0_cps_e3'][:]
        
        return np.array([e1, e2, e3])
    
    def read_proton_counts(self):
        """
        Read proton count channels (P1-P6)
        
        Returns
        -------
        proton_counts : ndarray, shape (6, n_time)
            Proton counts for channels P1, P2, P3, P4, P5, P6
        """
        p1 = self.ds_raw.variables['mep_pro_tel0_cps_p1'][:]
        p2 = self.ds_raw.variables['mep_pro_tel0_cps_p2'][:]
        p3 = self.ds_raw.variables['mep_pro_tel0_cps_p3'][:]
        p4 = self.ds_raw.variables['mep_pro_tel0_cps_p4'][:]
        p5 = self.ds_raw.variables['mep_pro_tel0_cps_p5'][:]
        p6 = self.ds_raw.variables['mep_pro_tel0_cps_p6'][:]
        
        return np.array([p1, p2, p3, p4, p5, p6])
    
    def read_time(self):
        """
        Read time array from raw file
        
        Returns
        -------
        time : ndarray
            Time in milliseconds since 1970 (epoch)
        """
        time_data = self.ds_raw.variables['time'][:]
        # Handle masked arrays - convert to regular array using unmasked data
        if isinstance(time_data, np.ma.MaskedArray):
            # If all masked, return the underlying data
            # If some masked, fill masked values with interpolation or forward fill
            if time_data.mask.all():
                # All masked - use underlying data
                return time_data.data.astype(np.float64)
            else:
                # Some masked - fill masked values
                time_filled = np.ma.filled(time_data, fill_value=np.nan)
                # Forward fill NaN values
                mask = np.isnan(time_filled)
                idx = np.where(~mask, np.arange(mask.size), 0)
                idx = np.maximum.accumulate(idx)
                time_filled = time_filled[idx]
                return time_filled.astype(np.float64)
        return time_data.astype(np.float64)
    
    def read_year_day(self):
        """
        Read year and day from raw file
        
        Returns
        -------
        year : ndarray
            Year (YYYY)
        day : ndarray
            Day of year (DDD)
        """
        year = self.ds_raw.variables['year'][:]
        day = self.ds_raw.variables['day'][:]
        
        return year, day
    
    def read_satellite_position(self):
        """
        Read satellite geographic position
        
        Returns
        -------
        lat : ndarray
            Geographic latitude (degrees)
        lon : ndarray
            Geographic longitude (degrees)
        """
        lat = self.ds_raw.variables['lat'][:]
        lon = self.ds_raw.variables['lon'][:]
        
        return lat, lon
    
    def read_footpoint_location(self):
        """
        Read field line footpoint location
        
        Returns
        -------
        fofl_lat : ndarray
            Footpoint geodetic latitude (degrees)
        fofl_lon : ndarray
            Footpoint geodetic longitude (degrees)
        """
        fofl_lat = self.ds_proc.variables['geod_lat_foot'][:]
        fofl_lon = self.ds_proc.variables['geod_lon_foot'][:]
        
        return fofl_lat, fofl_lon
    
    def read_magnetic_local_time(self):
        """
        Read magnetic local time (MLT)
        
        Returns
        -------
        mlt : ndarray
            MLT (degrees, 0-360)
        """
        return self.ds_proc.variables['MLT'][:]
    
    def read_l_value(self):
        """
        Read L-value (IGRF model)
        
        Returns
        -------
        l_value : ndarray
            L-value (unitless, typically 1-10)
        """
        return self.ds_proc.variables['L_IGRF'][:]
    
    def read_pitch_angle(self):
        """
        Read pitch angle at satellite
        
        Returns
        -------
        pitch_angle : ndarray
            Pitch angle (degrees, 0-180)
        """
        return self.ds_proc.variables['meped_alpha_0_sat'][:]
    
    def read_magnetic_field(self):
        """
        Read magnetic field strengths
        
        Returns
        -------
        b_sat : ndarray
            Magnetic field at satellite (Tesla)
        b_foot : ndarray
            Magnetic field at footpoint (Tesla)
        """
        b_sat = self.ds_proc.variables['Btot_sat'][:]
        b_foot = self.ds_proc.variables['Btot_foot'][:]
        
        return b_sat, b_foot
    
    def read_quality_flag(self):
        """
        Read data quality flag (IFC on/off)
        
        Returns
        -------
        ifc_flag : ndarray
            1 if instrument functioning, 0 otherwise
        """
        return self.ds_proc.variables['mep_IFC_on'][:]
    
    def read_all_data(self):
        """
        Read all data at once
        
        Returns
        -------
        dict
            Dictionary with all data arrays:
            {
                'electron_counts': (3, n_time),
                'proton_counts': (6, n_time),
                'time': (n_time,),
                'rtime': (n_time,),  # fractional hours since start of day
                'year': (n_time,),
                'day': (n_time,),
                'geog_lat': (n_time,),
                'geog_lon': (n_time,),
                'fofl_lat': (n_time,),
                'fofl_lon': (n_time,),
                'mlt': (n_time,),
                'l_value': (n_time,),
                'pitch_angle': (n_time,),
                'b_sat': (n_time,),
                'b_foot': (n_time,),
                'ifc_flag': (n_time,)
            }
        """
        time_data = self.read_time()
        year_data = self.ds_raw.variables['year'][:]
        day_data = self.ds_raw.variables['day'][:]
        
        # Calculate rtime (fractional hours since start of day)
        rtime_data = calculate_rtime(time_data, year_data, day_data)
        
        return {
            'electron_counts': self.read_electron_counts(),
            'proton_counts': self.read_proton_counts(),
            'time': time_data,
            'rtime': rtime_data,
            'year': year_data,
            'day': day_data,
            'geog_lat': self.ds_raw.variables['lat'][:],
            'geog_lon': self.ds_raw.variables['lon'][:],
            'fofl_lat': self.read_footpoint_location()[0],
            'fofl_lon': self.read_footpoint_location()[1],
            'mlt': self.read_magnetic_local_time(),
            'l_value': self.read_l_value(),
            'pitch_angle': self.read_pitch_angle(),
            'b_sat': self.read_magnetic_field()[0],
            'b_foot': self.read_magnetic_field()[1],
            'ifc_flag': self.read_quality_flag(),
        }
    
    def _compute_blc_angle(self, pitch_angle, b_sat, b_foot):
        """Compute Bounce Loss Cone (BLC) angle.
        
        IDL formula (from spectral_fits_00.pro):
            BLC_Alpha_temp = sqrt(B_sat / B_foot)
            BLC_alpha = asin(BLC_Alpha_temp)  [radians]
            BLC_Angle = BLC_alpha * 180 / π  [degrees]
        
        This represents the half-angle of the bounce loss cone in the equatorial plane.
        
        Parameters
        ----------
        pitch_angle : ndarray (unused - included for API compatibility)
            Pitch angle at satellite (degrees) - NOT used in IDL formula
        b_sat : ndarray
            Magnetic field at satellite (Tesla)
        b_foot : ndarray
            Magnetic field at footpoint (Tesla)
            
        Returns
        -------
        blc_angle : ndarray
            BLC angle (degrees)
        """
        # IDL formula: BLC_angle = asin(sqrt(B_sat / B_foot))

        with np.errstate(divide='ignore', invalid='ignore'):
            b_ratio = b_sat / b_foot
            # Compute sqrt of B ratio
            sqrt_b_ratio = np.sqrt(b_ratio)
            # Clamp to [0, 1] to avoid issues with arcsin
            sqrt_b_ratio = np.clip(sqrt_b_ratio, 0, 1.0)
        
        # Compute BLC angle in radians then convert to degrees
        blc_rad = np.arcsin(sqrt_b_ratio)
        blc_deg = np.rad2deg(blc_rad)
        
        # Set to NaN where inputs are invalid or zero
        invalid = ~(np.isfinite(b_sat) & np.isfinite(b_foot) & (b_sat > 0) & (b_foot > 0))
        blc_deg[invalid] = np.nan
        
        return blc_deg
    
    def _masked_average(self, arr_1d, n_bins):
        """Average array ignoring zero values (treat as missing data).
        
        For each bin of 8 values, computes the mean of non-zero values.
        If all 8 values in a bin are zero, returns zero.
        
        Parameters
        ----------
        arr_1d : ndarray
            1-D array trimmed to (n_bins * 8,)
        n_bins : int
            Number of output bins
            
        Returns
        -------
        avg : ndarray, shape (n_bins,)
            Averaged values with zeros ignored
        """
        reshaped = arr_1d.reshape(n_bins, 8)
        avg = np.zeros(n_bins, dtype=arr_1d.dtype)
        for i in range(n_bins):
            bin_vals = reshaped[i]
            non_zero = bin_vals[bin_vals != 0]
            if len(non_zero) > 0:
                avg[i] = non_zero.mean()
            else:
                avg[i] = 0.0
        return avg

    def average_to_16sec(self):
        """
        Average 2-second data to 16-second resolution (matching IDL code)
        
        This method averages all data by a factor of 8 (from 2-sec to 16-sec).
        It groups 8 consecutive 2-second samples and averages them.
        
        For magnetic field data (b_sat, b_foot), masked/NaN values are automatically
        ignored during nanmean averaging. 
        
        Returns
        -------
        dict
            Dictionary with averaged data arrays
        """
        all_data = self.read_all_data()
        
        n_samples = all_data['electron_counts'].shape[1]
        # Calculate number of 16-second bins (factor of 8 averaging)
        n_bins = n_samples // 8
        n_samples_trimmed = n_bins * 8  # Trim to multiple of 8
        
        # Trim data to multiple of 8
        e_counts_trimmed = all_data['electron_counts'][:, :n_samples_trimmed]
        p_counts_trimmed = all_data['proton_counts'][:, :n_samples_trimmed]
        time_trimmed = all_data['time'][:n_samples_trimmed]
        rtime_trimmed = all_data['rtime'][:n_samples_trimmed]
        mlt_trimmed = all_data['mlt'][:n_samples_trimmed]
        lat_trimmed = all_data['geog_lat'][:n_samples_trimmed]
        lon_trimmed = all_data['geog_lon'][:n_samples_trimmed]
        fofl_lat_trimmed = all_data['fofl_lat'][:n_samples_trimmed]
        fofl_lon_trimmed = all_data['fofl_lon'][:n_samples_trimmed]
        l_value_trimmed = all_data['l_value'][:n_samples_trimmed]
        pitch_trimmed = all_data['pitch_angle'][:n_samples_trimmed]
        b_sat_trimmed = all_data['b_sat'][:n_samples_trimmed]
        b_foot_trimmed = all_data['b_foot'][:n_samples_trimmed]
        
        # Convert masked arrays to regular arrays with NaN for masked values
        # This must be done BEFORE averaging to preserve data
        if isinstance(b_sat_trimmed, np.ma.MaskedArray):
            b_sat_trimmed = b_sat_trimmed.filled(np.nan)
        if isinstance(b_foot_trimmed, np.ma.MaskedArray):
            b_foot_trimmed = b_foot_trimmed.filled(np.nan)
        if isinstance(pitch_trimmed, np.ma.MaskedArray):
            pitch_trimmed = pitch_trimmed.filled(np.nan)
        
        # Reshape to (n_channels, n_bins, 8) and average over the 8 samples
        e_counts_avg = np.array([
            e_counts_trimmed[i].reshape(n_bins, 8).mean(axis=1)
            for i in range(3)
        ])
        
        p_counts_avg = np.array([
            p_counts_trimmed[i].reshape(n_bins, 8).mean(axis=1)
            for i in range(6)
        ])
        
        # For other parameters, average every 8 points
        time_avg = time_trimmed.reshape(n_bins, 8).mean(axis=1)
        rtime_avg = rtime_trimmed.reshape(n_bins, 8).mean(axis=1)
        mlt_avg = mlt_trimmed.reshape(n_bins, 8).mean(axis=1)
        lat_avg = lat_trimmed.reshape(n_bins, 8).mean(axis=1)
        lon_avg = lon_trimmed.reshape(n_bins, 8).mean(axis=1)
        fofl_lat_avg = fofl_lat_trimmed.reshape(n_bins, 8).mean(axis=1)
        fofl_lon_avg = fofl_lon_trimmed.reshape(n_bins, 8).mean(axis=1)
        l_value_avg = l_value_trimmed.reshape(n_bins, 8).mean(axis=1)
        pitch_avg = pitch_trimmed.reshape(n_bins, 8).mean(axis=1)
        
        # Use nanmean for magnetic field (automatically ignores NaN/masked values)
        b_sat_trimmed_shaped = b_sat_trimmed.reshape(n_bins, 8)
        b_foot_trimmed_shaped = b_foot_trimmed.reshape(n_bins, 8)
        b_sat_avg = np.nanmean(b_sat_trimmed_shaped, axis=1)
        b_foot_avg = np.nanmean(b_foot_trimmed_shaped, axis=1)
        
        # Compute BLC angle from averaged pitch, b_sat, and b_foot
        blc_angle_avg = self._compute_blc_angle(pitch_avg, b_sat_avg, b_foot_avg)
        
        return {
            'electron_counts': e_counts_avg,
            'proton_counts': p_counts_avg,
            'time': time_avg,
            'rtime': rtime_avg,
            'mlt': mlt_avg,
            'geog_lat': lat_avg,
            'geog_lon': lon_avg,
            'fofl_lat': fofl_lat_avg,
            'fofl_lon': fofl_lon_avg,
            'l_value': l_value_avg,
            'pitch_angle': pitch_avg,
            'b_sat': b_sat_avg,
            'b_foot': b_foot_avg,
            'blc_angle': blc_angle_avg,
        }
    
    def close(self):
        """Close both NetCDF datasets"""
        if self.ds_raw is not None:
            self.ds_raw.close()
        if self.ds_proc is not None:
            self.ds_proc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()