#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script_unfringe.py

Unfringe JWST/MIRI MRS s3d cubes located in `./data/`.

This script expects a *single* observation directory layout:

- `./data/` contains the four channel cubes (ch1..ch4)

It produces unfringed cubes alongside the originals with `_unfringed.fits`
appended.
"""

from __future__ import annotations

import argparse
import os
from astropy.io import fits

from functions import find_fits_files, unfringe_cube


# Default wavelength segment definitions adopted from the historical script.
LAM_SCS_BY_CHANNEL = {
    1: [[0, 5.7], [5.7, 6.6], [6.6, 100]],
    2: [[0, 8.7], [8.7, 10.1], [10.1, 100]],
    3: [[0, 13.4], [13.4, 15.5], [15.5, 100]],
    4: [[0, 20.8], [20.8, 24.25], [24.45, 100]],
}


def guess_channel_from_name(path: str) -> int | None:
    """Try to infer channel from filename."""
    low = os.path.basename(path).lower()
    for ch in (1, 2, 3, 4):
        if f"ch{ch}" in low:
            return ch
    return None


def process_one_file(path: str, channel: int, overwrite: bool = True) -> str:
    """Unfringe a single FITS cube and write `<name>_unfringed.fits`."""
    lam_scs = LAM_SCS_BY_CHANNEL[channel]
    hdul = fits.open(path)
    new_data = unfringe_cube(hdul, lam_scs, channel)
    hdul[1].data = new_data

    out_path = path.replace('.fits', '_unfringed.fits')
    hdul.writeto(out_path, overwrite=overwrite)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Unfringe MIRI MRS FITS cubes in ./data")
    parser.add_argument('--data-dir', default='./data', help='Directory containing the channel FITS cubes')
    parser.add_argument('--pattern', default='*.fits', help='Glob pattern to locate input FITS files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()

    files = find_fits_files(args.data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No FITS files found in {args.data_dir!r} with pattern {args.pattern!r}")

    # Only process known channels.
    processed = []
    for path in files:
        ch = guess_channel_from_name(path)
        if ch is None:
            continue
        processed.append(process_one_file(path, ch, overwrite=args.overwrite))

    if not processed:
        raise SystemExit(
            "Found FITS files, but none looked like MIRI channel files (expected 'ch1'...'ch4' in filename)."
        )

    print("Unfringed files:")
    for p in processed:
        print(" -", p)


if __name__ == '__main__':
    main()
