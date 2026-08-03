"""
Angular Corrections for MEPED Electron Detectors

This module implements corrections for MEPED electron telescope measurements
where the 0° (precipitating) telescope is measuring trapped electrons instead
of true precipitating electrons.

Based on Selesnick et al. (2020) JGR: Space Physics
"POES/MEPED Angular Response Functions and the Precipitating Radiation Belt Electron Flux"

The key finding: During low geomagnetic activity, the 0° telescope usually measures
trapped or quasi-trapped electrons, NOT precipitating electrons. This happens because
the telescope has non-zero sensitivity at high incidence angles where electron
intensity is much higher than in the loss cone.

Correction criterion:
- Low geomagnetic activity (Kp < 3)
- High latitude (|lat| > 50°)
- 90°/0° count ratio ≤ threshold (default: 2.0)
"""

import numpy as np
from datetime import datetime, timedelta
import os


# Constants
KP_FILE_PATH = os.path.join(os.path.dirname(__file__), 'kp_index.txt')
DEFAULT_KP_THRESHOLD = 3.0  # Kp < 3 indicates low activity
DEFAULT_RATIO_THRESHOLD = 2.0  # 90°/0° ratio below this is suspicious
DEFAULT_LATITUDE_THRESHOLD = 50.0  # degrees


def load_kp_data():
    """
    Load Kp index data from the text file.
    
    Returns
    -------
    kp_data : dict
        Dictionary with:
        - 'time': list of datetime objects (start of 3-hour intervals)
        - 'kp': list of Kp indices (float)
        - 'ap': list of Ap indices (int)
    """
    if not os.path.exists(KP_FILE_PATH):
        raise FileNotFoundError(f"Kp index file not found: {KP_FILE_PATH}")
    
    data = {
        'time': [],
        'kp': [],
        'ap': []
    }
    
    with open(KP_FILE_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 9:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = float(parts[3])  # Time in UT
                    
                    # Create datetime for the start of the 3-hour interval
                    dt = datetime(year, month, day, int(hour), int((hour % 1) * 60))
                    
                    kp = float(parts[7])  # Second to last column
                    ap = int(parts[8])   # Last column
                    
                    data['time'].append(dt)
                    data['kp'].append(kp)
                    data['ap'].append(ap)
                except (ValueError, IndexError):
                    continue
    
    return data


def get_kp_at_time(target_time, kp_data, method='closest'):
    """
    Get Kp index at a specific time using interpolation.
    
    Parameters
    ----------
    target_time : datetime or array-like of datetime
        Time(s) at which to get Kp index
    kp_data : dict
        Kp data from load_kp_data()
    method : str
        'closest' - use closest 3-hour interval
        'daily' - use daily average
        'linear' - linear interpolation between points
    
    Returns
    -------
    kp_values : float or array
        Kp index value(s) at target time(s)
    """
    if isinstance(target_time, datetime):
        single_time = True
        target_time = [target_time]
    else:
        single_time = False
    
    kp_times = np.array(kp_data['time'])
    kp_values = np.array(kp_data['kp'])
    
    # Pre-convert all times to timestamps for efficient comparison
    kp_timestamps = np.array([dt.timestamp() for dt in kp_times])
    
    if method == 'closest':
        # Find closest time index for each target time
        result = np.zeros(len(target_time))
        for i, t in enumerate(target_time):
            t_timestamp = t.timestamp()
            time_diffs = np.abs(kp_timestamps - t_timestamp)
            closest_idx = np.argmin(time_diffs)
            result[i] = kp_values[closest_idx]
        
    elif method == 'daily':
        # Calculate daily averages
        dates = np.array([dt.date() for dt in kp_times])
        unique_dates = np.unique(dates)
        daily_kp = {}
        
        for date in unique_dates:
            mask = dates == date
            daily_kp[date] = np.mean(kp_values[mask])
        
        # Get daily average for each target time
        result = np.zeros(len(target_time))
        for i, t in enumerate(target_time):
            date = t.date()
            if date in daily_kp:
                result[i] = daily_kp[date]
            else:
                # If date not found, use closest day
                # Convert dates to ordinal for comparison
                unique_dates_ordinal = np.array([d.toordinal() for d in unique_dates])
                target_ordinal = date.toordinal()
                date_diffs = np.abs(unique_dates_ordinal - target_ordinal)
                closest_date_idx = np.argmin(date_diffs)
                result[i] = daily_kp[unique_dates[closest_date_idx]]
                
    elif method == 'linear':
        # Linear interpolation (Kp is discrete, so this might not be ideal)
        from scipy.interpolate import interp1d
        time_seconds = np.array([t.timestamp() for t in kp_times])
        target_seconds = np.array([t.timestamp() for t in target_time])
        
        # Use nearest extrapolation for out-of-bounds
        interp_func = interp1d(time_seconds, kp_values, kind='nearest', 
                              fill_value='extrapolate')
        result = interp_func(target_seconds)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'closest', 'daily', or 'linear'.")
    
    return result[0] if single_time else result


def identify_contaminated_measurements(electron_counts_0deg, electron_counts_90deg,
                                       latitude, time_array, kp_data=None,
                                       kp_threshold=DEFAULT_KP_THRESHOLD,
                                       ratio_threshold=DEFAULT_RATIO_THRESHOLD,
                                       lat_threshold=DEFAULT_LATITUDE_THRESHOLD,
                                       kp_method='closest'):
    """
    Identify time periods where 0° telescope is likely measuring trapped electrons
    instead of precipitating electrons.
    
    Parameters
    ----------
    electron_counts_0deg : array
        Count rates from 0° (precipitating) telescope (counts/s or counts)
    electron_counts_90deg : array
        Count rates from 90° (trapped) telescope
    latitude : array
        Geographic or geomagnetic latitude (degrees)
    time_array : array of datetime
        Time points corresponding to the measurements
    kp_data : dict, optional
        Pre-loaded Kp data from load_kp_data(). If None, will load automatically.
    kp_threshold : float
        Maximum Kp index for "low activity" (default: 3.0)
    ratio_threshold : float
        Maximum 90°/0° ratio for valid precipitating measurements (default: 2.0)
    lat_threshold : float
        Minimum absolute latitude for high latitudes (default: 50.0)
    kp_method : str
        Method for interpolating Kp to measurement times: 'closest', 'daily', 'linear'
    
    Returns
    -------
    contamination_mask : bool array
        True where 0° measurements should be flagged as contaminated
    kp_values : array
        Kp index values at each measurement time (for reference)
    """
    # Convert inputs to numpy arrays
    electron_counts_0deg = np.asarray(electron_counts_0deg, dtype=float)
    electron_counts_90deg = np.asarray(electron_counts_90deg, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    
    # Load Kp data if not provided
    if kp_data is None:
        kp_data = load_kp_data()
    
    # Get Kp values at measurement times
    kp_values = get_kp_at_time(time_array, kp_data, method=kp_method)
    kp_values = np.asarray(kp_values, dtype=float)
    
    # Calculate ratio: 90° / 0° (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(electron_counts_0deg > 0, 
                       electron_counts_90deg / electron_counts_0deg, 
                       np.inf)
    
    # Identify contaminated periods
    is_low_activity = kp_values < kp_threshold
    is_high_latitude = np.abs(latitude) > lat_threshold
    is_suspicious_ratio = ratio <= ratio_threshold
    
    # Combined mask: contaminated measurements
    contamination_mask = is_low_activity & is_high_latitude & is_suspicious_ratio
    
    return contamination_mask, kp_values


class AngularCorrection:
    """
    Class to handle angular corrections for MEPED electron data.
    
    This implements an optional correction (default ON) that removes contaminated
    0° telescope measurements where trapped electrons are being measured instead
    of precipitating electrons.
    
    Parameters
    ----------
    enabled : bool
        Whether to apply the correction (default: True)
    kp_threshold : float
        Kp index threshold for low activity (default: 3.0)
    ratio_threshold : float
        Maximum 90°/0° ratio threshold (default: 2.0)
    lat_threshold : float
        Minimum latitude threshold (default: 50.0)
    kp_method : str
        Method for Kp interpolation: 'closest', 'daily', 'linear' (default: 'closest')
    """
    
    def __init__(self, enabled=True, kp_threshold=DEFAULT_KP_THRESHOLD,
                 ratio_threshold=DEFAULT_RATIO_THRESHOLD,
                 lat_threshold=DEFAULT_LATITUDE_THRESHOLD,
                 kp_method='closest'):
        self.enabled = enabled
        self.kp_threshold = kp_threshold
        self.ratio_threshold = ratio_threshold
        self.lat_threshold = lat_threshold
        self.kp_method = kp_method
        self._kp_data = None
    
    def load_kp_data(self):
        """Load Kp index data (lazy loading)"""
        if self._kp_data is None:
            self._kp_data = load_kp_data()
        return self._kp_data
    
    def apply_correction(self, electron_counts_0deg, electron_counts_90deg,
                        latitude, time_array, return_mask=False):
        """
        Apply angular correction to electron measurements.
        
        Parameters
        ----------
        electron_counts_0deg : array
            Count rates from 0° telescope
        electron_counts_90deg : array
            Count rates from 90° telescope
        latitude : array
            Latitude values (degrees)
        time_array : array of datetime
            Time points for the measurements
        return_mask : bool
            If True, also return the contamination mask
        
        Returns
        -------
        corrected_electron_counts_0deg : array
            0° electron counts with contaminated values set to 0
        contamination_mask : bool array (optional)
            Mask indicating which values were contaminated
        kp_values : array (optional)
            Kp index values at each time point
        """
        if not self.enabled:
            if return_mask:
                return electron_counts_0deg, np.zeros(len(electron_counts_0deg), dtype=bool), None
            return electron_counts_0deg
        
        # Identify contaminated measurements
        contamination_mask, kp_values = identify_contaminated_measurements(
            electron_counts_0deg, electron_counts_90deg, latitude, time_array,
            kp_data=self.load_kp_data(),
            kp_threshold=self.kp_threshold,
            ratio_threshold=self.ratio_threshold,
            lat_threshold=self.lat_threshold,
            kp_method=self.kp_method
        )
        
        # Apply correction: set contaminated values to 0
        corrected_counts = np.asarray(electron_counts_0deg, dtype=float).copy()
        contamination_mask = np.asarray(contamination_mask, dtype=bool)
        corrected_counts[contamination_mask] = 0.0
        
        if return_mask:
            return corrected_counts, contamination_mask, kp_values
        return corrected_counts
    
    def __repr__(self):
        status = "ENABLED" if self.enabled else "DISABLED"
        return (f"AngularCorrection({status}, kp_threshold={self.kp_threshold}, "
                f"ratio_threshold={self.ratio_threshold}, lat_threshold={self.lat_threshold}, "
                f"kp_method='{self.kp_method}')")