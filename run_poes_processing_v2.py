#!/usr/bin/env python3
"""
POES MPE Software
Author: Joshua Pettit, GMU/NASA GSFC

Features:
- Single file processing with date/satellite specification
- Bulk directory processing with automatic file detection
- Resolution options: 2-second (raw), 16-second (averaged)
- Progress reporting for long-running jobs
- Multi-telescope support: 00deg, 90deg, or both
- Selesnick angular correction (enabled by default when both telescopes are analyzed)
- Performance optimization options for faster processing

Usage Examples:
    # Single date processing
    python3 run_poes_processing_v2.py --date 2014-01-01 --satellite n15
    
    # Bulk directory processing
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15
    
    # Bulk with 2-second resolution
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --resolution 2sec
    
    # Using full file paths
    python3 run_poes_processing_v2.py --raw-file /path/poes_n15_20140101_raw.nc --proc-file /path/poes_n15_20140101_proc.nc --date 2014-01-01
    
    # Process only 0-degree telescope
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --telescope 00deg
    
    # Process only 90-degree telescope  
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --telescope 90deg
    
    # Process both telescopes with angular correction (default)
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --telescope both
    
    # Process both telescopes without angular correction
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --telescope both --no-angular-correction
    
    # Fast processing mode (2-3x faster)
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --fast
    
    # Custom optimization parameters
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --optimization-tolerance 1e-3 --max-optimization-iter 200
    
    # Default (sample files)
    python3 run_poes_processing_v2.py

Resolution Notes:
    - 'auto': Automatically detect based on data year (pre-2012: 16sec, post-2012: 2sec)
    - '2sec': Keep original 2-second resolution (post-2012 data only)
    - '16sec': Average to 16-second resolution (use for pre-2012 compatibility)
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Add MPE_Conversion to path
sys.path.insert(0, '/home/electron/eclipse-workspace/MPE_Conversion')

from data_io import POESRawDataReader, NetCDFWriter
from spectral_inversion_main import process_poes_data


def get_year_from_filename(filename):
    """Extract year from POES filename (format: poes_XXX_YYYYMMDD_*.nc)"""
    try:
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 3:
            date_str = parts[2]  # YYYYMMDD
            return int(date_str[:4])
    except:
        pass
    return None


def detect_resolution(raw_file):
    """
    Detect appropriate resolution based on file year
    Pre-2012 files are typically 16-second, post-2012 are 2-second
    
    Returns: '2sec' or '16sec'
    """
    year = get_year_from_filename(raw_file)
    if year and year >= 2012:
        return '2sec'
    else:
        return '16sec'


def process_single_file(raw_file, proc_file, output_dir, satellite, date_str, resolution='auto',
                       telescope='both', apply_angular_correction=None,
                       fast_mode=False, optimization_tolerance=1e-4, max_optimization_iter=500):
    """
    Process a single pair of POES files
    
    Parameters
    ----------
    raw_file : str
        Path to *_raw.nc file
    proc_file : str
        Path to *_proc.nc file
    output_dir : str
        Output directory for results
    satellite : str
        Satellite code (n15, n19, m01, etc.)
    date_str : str
        Date string (YYYY-MM-DD)
    resolution : str
        'auto', '2sec', or '16sec'
    telescope : str
        '00deg', '90deg', or 'both' (default)
    apply_angular_correction : bool or None
        None (default) = auto-enable when telescope='both'
        True = force enable, False = force disable
    fast_mode : bool
        Enable fast mode with reduced optimization parameters
    optimization_tolerance : float
        Optimization tolerance for spectral fitting
    max_optimization_iter : int
        Maximum iterations for spectral fitting
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    
    # Determine if we should apply angular correction
    if apply_angular_correction is None:
        apply_angular_correction = (telescope == 'both')
    
    # Validate angular correction usage
    if apply_angular_correction and telescope != 'both':
        print(f"\n❌ ERROR: Angular correction requires telescope='both', but got '{telescope}'")
        return False
    
    print(f"\n[Processing] {satellite.upper()} - {date_str} - Telescope: {telescope}")
    if apply_angular_correction:
        print("  [Angular Correction] ENABLED")
    else:
        print("  [Angular Correction] DISABLED")
    print("-" * 80)
    
    # Verify input files exist
    if not os.path.exists(raw_file):
        print(f"  ❌ ERROR: Raw file not found: {raw_file}")
        return False
    if not os.path.exists(proc_file):
        print(f"  ❌ ERROR: Proc file not found: {proc_file}")
        return False
    
    raw_size = os.path.getsize(raw_file) / 1e6
    proc_size = os.path.getsize(proc_file) / 1e6
    print(f"  Input files: {raw_size:.1f} MB + {proc_size:.1f} MB")
    
    # Detect resolution if auto
    if resolution == 'auto':
        resolution = detect_resolution(raw_file)
        print(f"  Auto-detected resolution: {resolution}")
    
    try:
        # Read data based on telescope selection
        with POESRawDataReader(raw_file, proc_file) as reader:
            
            # Handle resolution first
            if resolution == '16sec':
                # Average to 16-second resolution
                print(f"  Averaging to 16-second resolution...")
                
                if telescope == 'both':
                    # For both telescopes with 16sec resolution, we need to read and average telescope counts
                    # Angular correction will be applied INSIDE process_poes_data AFTER proton contamination removal
                    try:
                        # Read raw telescope data (2-second resolution)
                        all_telescope_counts = reader.read_all_telescope_counts()
                        
                        # Manually average telescope data to 16-second resolution
                        n_samples = all_telescope_counts['electron_0deg'].shape[1]
                        n_bins = n_samples // 8
                        n_samples_trimmed = n_bins * 8
                        
                        # Average electron counts for both telescopes
                        electron_0deg_avg = np.array([
                            all_telescope_counts['electron_0deg'][i, :n_samples_trimmed].reshape(n_bins, 8).mean(axis=1)
                            for i in range(3)
                        ])
                        electron_90deg_avg = np.array([
                            all_telescope_counts['electron_90deg'][i, :n_samples_trimmed].reshape(n_bins, 8).mean(axis=1)
                            for i in range(3)
                        ])
                        proton_0deg_avg = np.array([
                            all_telescope_counts['proton_0deg'][i, :n_samples_trimmed].reshape(n_bins, 8).mean(axis=1)
                            for i in range(6)
                        ])
                        
                        # Use averaged telescope data
                        electron_counts = electron_0deg_avg
                        electron_counts_90deg = electron_90deg_avg
                        proton_counts = proton_0deg_avg
                        
                        # Also need to average the other data from avg_data
                        avg_data = reader.average_to_16sec()
                        time = avg_data['time']
                        rtime = avg_data['rtime']
                        mlt = avg_data['mlt']
                        l_value = avg_data['l_value']
                        pitch_angle = avg_data['pitch_angle']
                        b_sat = avg_data['b_sat']
                        b_foot = avg_data['b_foot']
                        geog_lat = avg_data['geog_lat']
                        geog_lon = avg_data['geog_lon']
                        fofl_lat = avg_data['fofl_lat']
                        fofl_lon = avg_data['fofl_lon']
                        blc_angle = avg_data['blc_angle']
                        
                        resolution_label = "16-second"
                        
                    except Exception as e:
                        print(f"  ❌ ERROR preparing data for both telescopes at 16sec: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    # Single telescope at 16sec resolution
                    avg_data = reader.average_to_16sec()
                    proton_counts = avg_data['proton_counts']
                    electron_counts = avg_data['electron_counts']
                    time = avg_data['time']
                    rtime = avg_data['rtime']
                    mlt = avg_data['mlt']
                    l_value = avg_data['l_value']
                    pitch_angle = avg_data['pitch_angle']
                    b_sat = avg_data['b_sat']
                    b_foot = avg_data['b_foot']
                    geog_lat = avg_data['geog_lat']
                    geog_lon = avg_data['geog_lon']
                    fofl_lat = avg_data['fofl_lat']
                    fofl_lon = avg_data['fofl_lon']
                    blc_angle = avg_data['blc_angle']
                    resolution_label = "16-second"
            
            elif resolution == '2sec':
                # Use raw data without averaging
                print(f"  Using raw 2-second resolution...")
                
                if telescope == 'both':
                    # For both telescopes, we need to read all telescope counts
                    # Angular correction will be applied INSIDE process_poes_data AFTER proton contamination removal
                    try:
                        all_telescope_counts = reader.read_all_telescope_counts()
                        all_data = reader.read_all_data()
                        
                        time = all_data['time']
                        rtime = all_data['rtime']
                        mlt = all_data['mlt']
                        l_value = all_data['l_value']
                        pitch_angle = all_data['pitch_angle']
                        b_sat = all_data['b_sat']
                        b_foot = all_data['b_foot']
                        geog_lat = all_data['geog_lat']
                        geog_lon = all_data['geog_lon']
                        fofl_lat = all_data['fofl_lat']
                        fofl_lon = all_data['fofl_lon']
                        blc_angle = reader._compute_blc_angle(pitch_angle, b_sat, b_foot)
                        
                        # Use raw 0° electron data and 90° data, proton data from 0°
                        # Angular correction will be applied inside process_poes_data if enabled
                        electron_counts = all_telescope_counts['electron_0deg']
                        electron_counts_90deg = all_telescope_counts['electron_90deg']
                        proton_counts = all_telescope_counts['proton_0deg']
                        
                        resolution_label = "2-second"
                        
                    except Exception as e:
                        print(f"  ❌ ERROR preparing data for both telescopes: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    # Single telescope
                    all_data = reader.read_all_data()
                    proton_counts = all_data['proton_counts']
                    electron_counts = all_data['electron_counts']
                    time = all_data['time']
                    rtime = all_data['rtime']
                    mlt = all_data['mlt']
                    l_value = all_data['l_value']
                    pitch_angle = all_data['pitch_angle']
                    b_sat = all_data['b_sat']
                    b_foot = all_data['b_foot']
                    geog_lat = all_data['geog_lat']
                    geog_lon = all_data['geog_lon']
                    fofl_lat = all_data['fofl_lat']
                    fofl_lon = all_data['fofl_lon']
                    blc_angle = reader._compute_blc_angle(pitch_angle, b_sat, b_foot)
                    resolution_label = "2-second"
            else:
                print(f"  ❌ Unknown resolution: {resolution}")
                return False
            
            n_time = proton_counts.shape[1]
            print(f"  ✅ Processed to {n_time:,} time points ({resolution_label} resolution)")
    
    except Exception as e:
        print(f"  ❌ ERROR reading data: {e}")
        return False
    
    # Process data (spectral fitting)
    print(f"  Processing data through spectral inversion...")
    try:
        # Pass angular correction parameters if available
        kwargs = {
            'proton_counts': proton_counts,
            'electron_counts': electron_counts,
            'date_str': date_str,
            'satellite': satellite
        }
        
        # Add performance optimization parameters
        # If fast_mode is True, use aggressive optimization settings
        final_optimization_tolerance = 1e-3 if fast_mode else optimization_tolerance
        final_max_optimization_iter = 200 if fast_mode else max_optimization_iter
        
        kwargs.update({
            'optimization_tolerance': final_optimization_tolerance,
            'max_optimization_iter': final_max_optimization_iter
        })
        
        # Add angular correction parameters if we have both telescopes
        if apply_angular_correction and 'electron_counts_90deg' in locals():
            import pandas as pd
            from datetime import datetime
            time_datetime = pd.to_datetime(time, unit='ms')
            # Convert pandas Timestamp to datetime objects
            time_array_for_angular = [ts.to_pydatetime() for ts in time_datetime]
            kwargs.update({
                'electron_counts_90deg': electron_counts_90deg,
                'apply_angular_correction': True,
                'latitude': geog_lat,
                'time_array': time_array_for_angular
            })
        
        results = process_poes_data(**kwargs)
        
        print(f"  ✅ Spectral fitting complete!")
    
    except Exception as e:
        print(f"  ❌ ERROR processing data: {e}")
        return False
    
    # Write output file
    try:
        # Create output filename with resolution indicator
        res_str = '16s' if '16sec' in resolution else '2s'
        # Add telescope suffix to output filename
        telescope_suffix = f'_{telescope}' if telescope != 'both' else '_both'
        
        output_file = os.path.join(
            output_dir, 
            f'POES_flux_{satellite}_{date_str.replace("-", "")}_{res_str}{telescope_suffix}.nc'
        )
        
        writer = NetCDFWriter(output_file)
        writer.create_file(
            energy=results['energy'],
            n_time_steps=proton_counts.shape[1],
            n_proton_channels=6,
            n_electron_channels=4
        )
        # Attach provenance of response CSVs to global attributes
        resp_paths = results.get('response_paths', {})
        try:
            if resp_paths.get('electron_electron'):
                writer.ds.setncattr('response_electron_electron', resp_paths.get('electron_electron'))
            if resp_paths.get('electron_proton'):
                writer.ds.setncattr('response_electron_proton', resp_paths.get('electron_proton'))
            if resp_paths.get('proton_proton'):
                writer.ds.setncattr('response_proton_proton', resp_paths.get('proton_proton'))
        except Exception:
            pass

        print(f"  Writing output in batch...")
        writer.write_data_batch({
            'time':                     time,
            'rtime':                    rtime,
            'geog_lat':                 geog_lat,
            'geog_lon':                 geog_lon,
            'fofl_lat':                 fofl_lat,
            'fofl_lon':                 fofl_lon,
            'mlt':                      mlt,
            'lvalue':                   l_value,
            'pitch':                    pitch_angle,
            'bfofl':                    b_foot,
            'blocal':                   b_sat,
            'blc_angle':                blc_angle,
            'electron_flux':            results['electron_flux'],        # (27, n_time)
            'proton_flux':              results['proton_flux'],           # (27, n_time)
            'electron_counts':          electron_counts,                  # (3, n_time)
            'proton_counts':            proton_counts,                    # (6, n_time)
            'proton_contamination':     results.get('proton_contamination'),   # (3, n_time)
            'corrected_electron_counts': results.get('corrected_electron_counts'),  # (4, n_time)
        })
        writer.close()
        
        output_size = os.path.getsize(output_file) / 1e6
        print(f"  ✅ Output: {os.path.basename(output_file)} ({output_size:.1f} MB)")
        return True
    
    except Exception as e:
        print(f"  ❌ ERROR writing output: {e}")
        return False


def process_directory(directory, satellite, output_dir=None, resolution='auto',
                     telescope='both', no_angular_correction=False,
                     fast_mode=False, optimization_tolerance=1e-4, max_optimization_iter=500):
    """
    Process all POES files in a directory
    
    Parameters
    ----------
    directory : str
        Directory containing poes_*_raw.nc and poes_*_proc.nc files
    satellite : str
        Satellite code (will filter files)
    output_dir : str, optional
        Output directory (defaults to input directory)
    resolution : str
        'auto', '2sec', or '16sec'
    telescope : str
        '00deg', '90deg', or 'both' (default)
    no_angular_correction : bool
        Whether to disable angular correction (default: False)
    fast_mode : bool
        Enable fast mode with reduced optimization parameters
    optimization_tolerance : float
        Optimization tolerance for spectral fitting
    max_optimization_iter : int
        Maximum iterations for spectral fitting
    
    Returns
    -------
    dict
        Statistics on processed files
    """
    
    if output_dir is None:
        output_dir = directory
    
    print("\n" + "="*80)
    print("BULK PROCESSING - POES DATA DIRECTORY")
    print("="*80)
    print(f"Input directory:  {directory}")
    print(f"Output directory: {output_dir}")
    print(f"Satellite filter: {satellite}")
    print(f"Resolution:       {resolution}")
    print("="*80)
    
    # Find all raw files for this satellite
    pattern = os.path.join(directory, f'poes_{satellite}_*_raw.nc')
    raw_files = sorted(glob.glob(pattern))
    
    if not raw_files:
        print(f"\n❌ No files found matching: {pattern}")
        return {'total': 0, 'successful': 0, 'failed': 0, 'files': []}
    
    print(f"\nFound {len(raw_files)} file(s) to process:\n")
    
    stats = {
        'total': len(raw_files),
        'successful': 0,
        'failed': 0,
        'files': []
    }
    
    for i, raw_file in enumerate(raw_files, 1):
        # Extract satellite and date from filename
        basename = os.path.basename(raw_file)
        # Format: poes_XXX_YYYYMMDD_raw.nc
        parts = basename.split('_')
        if len(parts) < 3:
            print(f"[{i}/{len(raw_files)}] ❌ Invalid filename: {basename}")
            stats['failed'] += 1
            continue
        
        date_str_raw = parts[2]  # YYYYMMDD
        # Convert to YYYY-MM-DD
        date_str = f"{date_str_raw[0:4]}-{date_str_raw[4:6]}-{date_str_raw[6:8]}"
        sat_code = parts[1]
        
        # Find corresponding proc file
        proc_file = os.path.join(directory, f'poes_{sat_code}_{date_str_raw}_proc.nc')
        
        if not os.path.exists(proc_file):
            print(f"[{i}/{len(raw_files)}] ❌ Missing proc file for {date_str}")
            stats['failed'] += 1
            continue
        
        # Process this file
        success = process_single_file(
            raw_file=raw_file,
            proc_file=proc_file,
            output_dir=output_dir,
            satellite=sat_code,
            date_str=date_str,
            resolution=resolution,
            telescope=telescope,
            apply_angular_correction=not no_angular_correction,
            fast_mode=fast_mode,
            optimization_tolerance=optimization_tolerance,
            max_optimization_iter=max_optimization_iter
        )
        
        if success:
            stats['successful'] += 1
            stats['files'].append({'date': date_str, 'satellite': sat_code, 'status': 'success'})
        else:
            stats['failed'] += 1
            stats['files'].append({'date': date_str, 'satellite': sat_code, 'status': 'failed'})
    
    # Print summary
    print("\n" + "="*80)
    print("BULK PROCESSING SUMMARY")
    print("="*80)
    print(f"Total:       {stats['total']}")
    print(f"Successful:  {stats['successful']} ✅")
    print(f"Failed:      {stats['failed']} ❌")
    if stats['total'] > 0:
        success_rate = int(100 * stats['successful'] / stats['total'])
        print(f"Success rate: {success_rate}%")
    print("="*80 + "\n")
    
    return stats


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='POES Spectral Flux Inversion - Process satellite data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  
  Single date (auto-detect resolution):
    python3 run_poes_processing_v2.py --date 2014-01-01 --satellite n15
  
  Single date with specific resolution:
    python3 run_poes_processing_v2.py --date 2019-09-06 --satellite n19 --resolution 2sec
  
  Bulk directory processing (auto-detect each file):
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15
  
  Bulk with forced 2-second resolution:
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --resolution 2sec
  
  Bulk with forced 16-second resolution:
    python3 run_poes_processing_v2.py --directory /path/to/data --satellite n15 --resolution 16sec
  
  Full file paths:
    python3 run_poes_processing_v2.py --raw-file /path/raw.nc --proc-file /path/proc.nc --date 2014-01-01

Note: Resolution auto-detection uses year (pre-2012: 16sec, post-2012: 2sec)
        '''
    )
    
    parser.add_argument('--date', type=str, default=None,
                       help='Process single date (YYYY-MM-DD)')
    parser.add_argument('--directory', type=str, default=None,
                       help='Bulk process entire directory')
    parser.add_argument('--satellite', type=str, default='n15',
                       help='Satellite code (n15, n18, n19, m01, m02, etc.)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (defaults to input directory)')
    parser.add_argument('--resolution', type=str, choices=['auto', '2sec', '16sec'],
                       default='auto',
                       help='Resolution: auto-detect (default), 2sec (raw), or 16sec (averaged)')
    parser.add_argument('--raw-file', type=str, default=None,
                       help='Full path to raw.nc file')
    parser.add_argument('--proc-file', type=str, default=None,
                       help='Full path to proc.nc file')
    
    # Telescope selection and angular correction
    parser.add_argument('--telescope', type=str, choices=['00deg', '90deg', 'both'],
                       default='both',
                       help='Which telescope(s) to process: 00deg, 90deg, or both (default)')
    parser.add_argument('--no-angular-correction', action='store_true', default=False,
                       help='Disable Selesnick angular correction (only applicable when telescope=both)')
    
    # Performance optimization arguments
    parser.add_argument('--fast', action='store_true', default=False,
                       help='Enable fast mode: reduces optimization iterations and tolerance for 2-3x speedup')
    parser.add_argument('--optimization-tolerance', type=float, default=1e-4,
                       help='Optimization tolerance (default: 1e-4, use 1e-3 for faster but less precise results)')
    parser.add_argument('--max-optimization-iter', type=int, default=500,
                       help='Maximum optimization iterations (default: 500, reduce to 200-300 for faster processing)')
    
    args = parser.parse_args()
    
    # Determine processing mode
    # --raw-file/--proc-file always takes priority when both are explicitly given
    if args.raw_file and args.proc_file:
        # Full path processing
        date_str = args.date or '2017-12-22'
        output_dir = args.output or os.path.dirname(args.raw_file)

        success = process_single_file(
            raw_file=args.raw_file,
            proc_file=args.proc_file,
            output_dir=output_dir,
            satellite=args.satellite,
            date_str=date_str,
            telescope=args.telescope,
            apply_angular_correction=not args.no_angular_correction,
            resolution=args.resolution,
            fast_mode=args.fast,
            optimization_tolerance=args.optimization_tolerance,
            max_optimization_iter=args.max_optimization_iter
        )
        return success

    elif args.directory:
        # Bulk processing
        stats = process_directory(
            directory=args.directory,
            satellite=args.satellite,
            output_dir=args.output,
            resolution=args.resolution,
            telescope=args.telescope,
            no_angular_correction=args.no_angular_correction,
            fast_mode=args.fast,
            optimization_tolerance=args.optimization_tolerance,
            max_optimization_iter=args.max_optimization_iter
        )
        return stats['successful'] == stats['total']
    
    elif args.date:
        # Single date processing
        date_str = args.date
        date_parts = date_str.replace('-', '')  # YYYYMMDD
        
        # Default file locations
        data_dir = args.output or '/home/electron/Downloads/'
        raw_file = os.path.join(data_dir, f'poes_{args.satellite}_{date_parts}_raw.nc')
        proc_file = os.path.join(data_dir, f'poes_{args.satellite}_{date_parts}_proc.nc')
        
        success = process_single_file(
            raw_file=raw_file,
            proc_file=proc_file,
            output_dir=data_dir,
            satellite=args.satellite,
            date_str=date_str,
            resolution=args.resolution,
            telescope=args.telescope,
            apply_angular_correction=not args.no_angular_correction,
            fast_mode=args.fast,
            optimization_tolerance=args.optimization_tolerance,
            max_optimization_iter=args.max_optimization_iter
        )
        return success

    else:
        # Default configuration (sample files)
        print("\n" + "="*80)
        print("POES SPECTRAL FLUX INVERSION - PROCESSING PIPELINE")
        print("="*80)
        print("\nNo arguments provided. Using sample files...\n")
        
        raw_file = '/home/electron/Downloads/poes_n15_20140101_raw.nc'
        proc_file = '/home/electron/Downloads/poes_n15_20140101_proc.nc'
        output_dir = '/home/electron/Downloads/'
        satellite = 'n15'
        date_str = '2014-01-01'
        resolution = 'auto'
        
        success = process_single_file(
            raw_file=raw_file,
            proc_file=proc_file,
            output_dir=output_dir,
            satellite=satellite,
            date_str=date_str,
            resolution=resolution,
            telescope='both',  # Default for sample files
            apply_angular_correction=True,
            fast_mode=False,  # Use default optimization for sample files
            optimization_tolerance=1e-4,
            max_optimization_iter=500
        )
        return success


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
