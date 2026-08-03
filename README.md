# POES MPE Spectral Inversion Software

A Python implementation for processing and analyzing particle flux data from POES (Polar Orbiting Environmental Satellites) MEPED (Medium Energy Proton and Electron Detector) instruments. This software performs spectral inversion to derive electron and proton fluxes from satellite count rates, with optional angular corrections for contaminated measurements.

## Features

- **Spectral Inversion**: Convert raw satellite count rates to physical particle fluxes
- **Multiple Spectral Models**: Support for Relativistic Maxwellian, Power Law, Energy Exponential, and Double Relativistic Maxwellian spectra
- **Angular Correction**: Selesnick correction for contaminated 0° telescope measurements during low geomagnetic activity
- **Proton Contamination Removal**: Estimate and remove proton contamination from electron channels
- **Multi-Resolution Support**: Process data at 2-second (raw) or 16-second (averaged) resolution
- **Multi-Telescope Support**: Process 0°, 90°, or both telescopes simultaneously
- **Bulk Processing**: Process single files or entire directories automatically
- **NetCDF Output**: Write processed data to standardized NetCDF files

## Installation

### Prerequisites

- Python 3.8+
- Required packages: `numpy`, `scipy`, `netCDF4`, `matplotlib`

### Install Dependencies

```bash
pip install numpy scipy netCDF4 matplotlib
```

### Optional Dependencies

- `spacepy` (for CDF file support)
- `xarray` (for advanced NetCDF operations)

## Quick Start

### Basic Usage

Process a single day of data for a specific satellite:

```bash
python run_poes_processing_v2.py --date 2014-01-01 --satellite n15
```

### Process a Directory

Process all files in a directory:

```bash
python run_poes_processing_v2.py --directory /path/to/data --satellite n15
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--date` | Date to process (YYYY-MM-DD) | Current date |
| `--satellite` | Satellite identifier (n15, n16, n17, n18, n19) | n15 |
| `--directory` | Directory containing input files | Current directory |
| `--resolution` | Data resolution: '2sec', '16sec', or 'auto' | auto |
| `--telescope` | Telescope(s) to process: '00deg', '90deg', or 'both' | both |
| `--no-angular-correction` | Disable Selesnick angular correction | Enabled |
| `--fast` | Enable fast processing mode (2-3x faster) | Disabled |
| `--optimization-tolerance` | Optimization convergence tolerance | 1e-4 |
| `--max-optimization-iter` | Maximum optimization iterations | 500 |

## Core Modules

### `spectral_inversion_main.py`

Main processing module containing the `POESSpectralInversion` class that orchestrates the entire spectral inversion process.

**Key Features:**
- Forward modeling of count rates from flux spectra
- Optimization-based spectral fitting
- Proton contamination estimation and removal
- Weighted flux calculation across multiple spectral models

**Supported Spectral Models:**
- `'relmaxwell'`: Relativistic Maxwellian
- `'powerlaw'`: Power Law distribution
- `'exponential'`: Energy Exponential
- `'drelmaxwell'`: Double Relativistic Maxwellian

### `angular_corrections.py`

Implements the Selesnick angular correction for contaminated electron telescope measurements.

**Correction Criteria:**
- Low geomagnetic activity (Kp < 3)
- High latitude (\|lat\| > 50°)
- 90°/0° count ratio ≤ 2.0

During these conditions, the 0° telescope may measure trapped electrons instead of true precipitating electrons, leading to contaminated measurements.

### `spectral_functions.py`

Contains the mathematical implementations of various spectral functions used for fitting particle flux data.

### `geometric_factors.py`

Handles geometric factor calculations and response function processing for POES instruments.

### `penalty_functions.py`

Implements penalty functions and optimization utilities for spectral fitting.

### `data_io.py`

Handles data input/output operations, including NetCDF and CDF file support.

## Python API Usage

### Basic Processing

```python
from spectral_inversion_main import POESSpectralInversion
import numpy as np

# Initialize processor
processor = POESSpectralInversion(
    min_energy=25.0,
    max_proton_energy=10000.0,
    max_electron_energy=10000.0,
    interp_level=27,
    dy=0.4,  # 40% relative error
    dt=16.0  # 16-second integration time
)

# Example proton counts (5 channels)
proton_counts = np.array([100, 200, 150, 80, 40])  # P1-P5 counts

# Fit proton spectrum
proton_results = processor.fit_proton_spectrum(proton_counts)
print(f"Best fit model: {proton_results['best_model']}")
print(f"Fit parameters: {proton_results['best_params']}")

# Get proton flux
proton_flux = processor.get_proton_flux_from_fit(
    proton_results['best_params'], 
    proton_results['best_model']
)
```

### With Angular Correction

```python
from angular_corrections import AngularCorrection
import numpy as np
from datetime import datetime

# Initialize angular correction
angular_corr = AngularCorrection(
    enabled=True,
    kp_threshold=3.0,
    ratio_threshold=2.0,
    lat_threshold=50.0
)

# Example data
electron_counts_0deg = np.array([1000, 1200, 800])  # 0° telescope
electron_counts_90deg = np.array([500, 600, 400])  # 90° telescope
latitudes = np.array([60.0, -70.0, 75.0])
times = [datetime(2015, 9, 7, 0, 10), datetime(2015, 9, 7, 1, 30), datetime(2015, 9, 7, 3, 45)]

# Apply correction
corrected_counts = angular_corr.apply_correction(
    electron_counts_0deg, electron_counts_90deg, latitudes, times, return_mask=True
)
```

## Data Requirements

### Input Files

The software expects POES MEPED data in either:
- NetCDF format (preferred)
- CDF format (legacy)

Input files should contain:
- Count rates for electron and proton channels
- Time stamps
- Geographic and geomagnetic coordinates
- Pitch angles
- L-values

### Required Channels

**Proton Channels (P1-P5):**
- P1: ~39 keV
- P2: ~115 keV
- P3: ~332 keV
- P4: ~1105 keV
- P5: ~2723 keV

**Electron Channels (E1-E4):**
- E1: ~72 keV
- E2: ~193 keV
- E3: ~419 keV
- E4: ~879 keV

### Kp Index Data

The angular correction requires Kp index data in `kp_index.txt` format. This file should contain 3-hour Kp indices with columns:
- Year, Month, Day, Hour (UT)
- Kp index (second to last column)
- Ap index (last column)

## Output Files

Processed data is written to NetCDF files with the following naming convention:
- `POES_flux_{satellite}_{date}_{resolution}_{telescopes}.nc`

Output variables include:
- `Ecounts`, `Pcounts`: Original count rates
- `CorrectedEOcounts`: Electron counts with corrections applied
- `ERMq`, `EPLq`, `EEEq`, `EDMq`: Fitted spectral parameters
- `ProtonContamination`: Estimated proton contamination in electron channels
- Coordinate variables: `time`, `geogLat`, `geogLon`, `foflLat`, `foflLon`, `MLT`, `lValue`, `pitch`, etc.

## Configuration

### Spectral Inversion Parameters

```python
processor = POESSpectralInversion(
    min_energy=25.0,           # Minimum energy (keV)
    max_proton_energy=10000.0, # Maximum proton energy (keV)
    max_electron_energy=10000.0, # Maximum electron energy (keV)
    interp_level=27,           # Number of interpolation points
    dy=0.4,                   # Relative measurement error (40%)
    dt=16.0,                  # Integration time (seconds)
    optimization_tolerance=1e-4,  # Optimization convergence tolerance
    max_optimization_iter=500   # Maximum optimization iterations
)
```

### Angular Correction Parameters

```python
angular_corr = AngularCorrection(
    enabled=True,             # Enable/disable correction
    kp_threshold=3.0,         # Kp index threshold for low activity
    ratio_threshold=2.0,      # 90°/0° count ratio threshold
    lat_threshold=50.0,       # Latitude threshold (degrees)
    kp_method='closest'       # Kp interpolation method: 'closest', 'daily', 'linear'
)
```

## Examples

### Example 1: Single File Processing

```bash
python run_poes_processing_v2.py \
    --date 2015-09-07 \
    --satellite n15 \
    --telescope both \
    --resolution 16sec
```

### Example 2: Bulk Processing with Angular Correction Disabled

```bash
python run_poes_processing_v2.py \
    --directory /path/to/raw_data \
    --satellite n15 \
    --telescope both \
    --no-angular-correction
```

### Example 3: Fast Processing with Custom Parameters

```bash
python run_poes_processing_v2.py \
    --directory /path/to/data \
    --satellite n15 \
    --fast \
    --optimization-tolerance 1e-3 \
    --max-optimization-iter 200
```

## Scientific Background

### Spectral Inversion

The spectral inversion process converts instrument count rates to physical particle fluxes by solving the inverse problem:

```
C = K^T @ f(E) + ε
```

Where:
- `C` is the vector of count rates
- `K` is the response matrix (geometric factors)
- `f(E)` is the differential flux as a function of energy
- `ε` is the measurement error

### Selesnick Angular Correction

Based on Selesnick et al. (2020) JGR: Space Physics, "POES/MEPED Angular Response Functions and the Precipitating Radiation Belt Electron Flux"

**Key Finding:** During low geomagnetic activity (Kp < 3), the 0° telescope can measure trapped electrons instead of true precipitating electrons due to its non-zero sensitivity at high incidence angles where electron intensity is much higher.

**Correction:** When the criteria are met (Kp < 3, |lat| > 50°, 90°/0° ratio ≤ 2.0), the contaminated 0° measurements are flagged and corrected by setting them to zero.

## File Structure

```
MPE_v2_v1/
├── angular_corrections.py      # Selesnick angular correction implementation
├── spectral_inversion_main.py # Main spectral inversion processor
├── spectral_functions.py       # Spectral model implementations
├── geometric_factors.py        # Geometric factor calculations
├── penalty_functions.py        # Optimization penalty functions
├── data_io.py                  # Data input/output operations
├── run_poes_processing_v2.py   # Main execution script
├── modified_bessel.py          # Modified Bessel function implementations
├── kp_index.txt                # Kp index data file
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **Missing Kp Index Data**: Ensure `kp_index.txt` is in the working directory
2. **File Not Found**: Check file paths and permissions
3. **Missing Dependencies**: Install required packages with `pip install numpy scipy netCDF4`
4. **Slow Processing**: Use `--fast` flag or reduce `--max-optimization-iter`

### Debug Mode

For detailed debugging information, modify the logging level in the main processing script.

## Citation

If you use this software in your research, please cite:

- Selesnick, R. S., Baker, D. N., Jaynes, A. N., et al. (2020). POES/MEPED Angular Response Functions and the Precipitating Radiation Belt Electron Flux. Journal of Geophysical Research: Space Physics, 125(4), e2019JA027414.

## License

This software is provided for research purposes. Contact the authors for licensing information.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## Support

For questions or issues, please contact the development team or open an issue on GitHub.

---

**Acknowledgments:** This software was developed at NASA GSFC in collaboration with George Mason University, based on the original IDL implementation by R. Selesnick.
