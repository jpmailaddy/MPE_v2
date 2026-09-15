#!/usr/bin/env python3
"""
Standalone script to compute POES MPE Bounce Loss Cone (BLC) flux for a
batch of files.

Given a directory of 0-degree telescope NetCDF files and a directory of
90-degree telescope NetCDF files (as written by run_poes_processing_v2.py /
data_io.py), this pairs up matching files by satellite+date+cadence and
writes one BLC output NetCDF file per pair.

Expected filename convention (matches POES_flux_m01_20250101_16s_00deg.nc):
    POES_flux_<sat>_<YYYYMMDD>_<cadence>_00deg.nc
    POES_flux_<sat>_<YYYYMMDD>_<cadence>_90deg.nc

Usage
-----
    python run_blc_batch.py \
        --dir0 /path/to/00deg_files \
        --dir90 /path/to/90deg_files \
        --outdir /path/to/blc_output

Optional:
    --dry-run     List the pairs that would be processed without actually
                  running the calculation.
"""

import argparse
import os
import re
import sys

# Import the BLC calculation logic. Adjust this import if the module
# containing calculate_blc_for_day lives elsewhere in your project.
from calculateBLC_v3 import calculate_blc_for_day


# Matches "POES_flux_m01_20250101_16s_00deg.nc" / "..._90deg.nc" and captures
# everything before the "_00deg"/"_90deg" suffix as the pairing key
# (e.g. "POES_flux_m01_20250101_16s"), plus which angle it is.
FILENAME_RE = re.compile(
    r"^(?P<key>.+)_(?P<angle>00|90)deg\.(nc|cdf|netcdf)$",
    re.IGNORECASE,
)


def _index_directory(directory, expected_angle):
    """Map pairing key -> full file path for every matching NetCDF file in a directory."""
    index = {}
    for name in sorted(os.listdir(directory)):
        match = FILENAME_RE.match(name)
        if match is None:
            continue
        angle = match.group('angle')
        if angle != expected_angle:
            # e.g. a 90deg file accidentally sitting in the 00deg directory
            print(
                f"Warning: '{name}' in {directory} looks like a {angle}deg file, "
                f"not {expected_angle}deg -- skipping.",
                file=sys.stderr,
            )
            continue
        key = match.group('key')
        if key in index:
            print(
                f"Warning: duplicate key '{key}' in {directory} "
                f"('{index[key]}' and '{name}') -- keeping the first match.",
                file=sys.stderr,
            )
            continue
        index[key] = os.path.join(directory, name)

    if not index:
        print(f"Warning: no *_{expected_angle}deg.nc files found in {directory}.", file=sys.stderr)

    return index


def find_pairs(dir0, dir90):
    """Match 00-degree and 90-degree files by their shared satellite/date/cadence key."""
    index0 = _index_directory(dir0, '00')
    index90 = _index_directory(dir90, '90')

    keys0 = set(index0)
    keys90 = set(index90)

    common = sorted(keys0 & keys90)
    only0 = sorted(keys0 - keys90)
    only90 = sorted(keys90 - keys0)

    if only0:
        print(f"Warning: {len(only0)} file(s) in {dir0} had no 90-deg match: {only0}", file=sys.stderr)
    if only90:
        print(f"Warning: {len(only90)} file(s) in {dir90} had no 00-deg match: {only90}", file=sys.stderr)

    return [(key, index0[key], index90[key]) for key in common]


def run(dir0, dir90, outdir, dry_run=False):
    os.makedirs(outdir, exist_ok=True)
    pairs = find_pairs(dir0, dir90)

    if not pairs:
        print("No matching 00-degree/90-degree file pairs found.", file=sys.stderr)
        return 1

    print(f"Found {len(pairs)} matching pair(s).")

    n_ok, n_failed = 0, 0
    for key, file_0deg, file_90deg in pairs:
        out_path = os.path.join(outdir, f"{key}_BLC.nc")

        if dry_run:
            print(f"[dry-run] {file_0deg}  +  {file_90deg}  ->  {out_path}")
            continue

        print(f"Processing '{key}':\n  00deg: {file_0deg}\n  90deg: {file_90deg}\n  -> {out_path}")
        try:
            calculate_blc_for_day(file_0deg, file_90deg, output_file=out_path)
            n_ok += 1
        except Exception as exc:
            print(f"  ERROR processing '{key}': {exc}", file=sys.stderr)
            n_failed += 1

    if not dry_run:
        print(f"Done. {n_ok} succeeded, {n_failed} failed.")
        if n_failed:
            return 1
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute POES MPE Bounce Loss Cone (BLC) flux for a batch of "
            "00deg/90deg file pairs (e.g. POES_flux_m01_20250101_16s_00deg.nc / "
            "POES_flux_m01_20250101_16s_90deg.nc)."
        )
    )
    parser.add_argument('--dir0', required=True, help="Directory containing *_00deg.nc telescope NetCDF files.")
    parser.add_argument('--dir90', required=True, help="Directory containing *_90deg.nc telescope NetCDF files.")
    parser.add_argument('--outdir', required=True, help="Directory to write BLC output NetCDF files into.")
    parser.add_argument('--dry-run', action='store_true', help="List pairs without processing them.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args.dir0, args.dir90, args.outdir, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
