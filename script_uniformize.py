#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script_uniformize.py

Uniformize (resample + merge) JWST/MIRI MRS FITS cubes.

This script expects a single observation stored in `./data/` containing the
four channel cubes (ideally already unfringed).

It produces one output cube in `./output/`.

Important
---------
The merge recipe is inherited from the original project and assumes the
"split-into-segments" routine yields 10 segments. If your 4-channel set differs, you may need to generalize
`perform_linear_average`.
"""

from __future__ import annotations

import argparse
import os

from astropy.io import fits

from functions import (
    extract_data_from_hdu,
    find_fits_files,
    interpolate_data,
    open_and_sort_fits_files,
    perform_linear_average,
    check_spatial_misalignment,
    update_header_with_wcs,
)


def main():
    parser = argparse.ArgumentParser(description="Uniformize (resample+merge) MIRI MRS cubes")
    parser.add_argument('--data-dir', default='./data', help='Directory containing the channel cubes')
    parser.add_argument('--pattern', default='*unfringed.fits', help='Glob pattern for input cubes')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--output-name', default='uniform_cube_unfringed.fits', help='Output FITS filename')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = find_fits_files(args.data_dir, args.pattern)
    if not filenames:
        raise SystemExit(f"No FITS files found in {args.data_dir!r} with pattern {args.pattern!r}")

    hdus, _wls = open_and_sort_fits_files(filenames)

    ras, decs, lams, datas = [], [], [], []
    for hdu in hdus:
        ra, dec, lam, data = extract_data_from_hdu(hdu)
        ras.append(ra)
        decs.append(dec)
        lams.append(lam)
        datas.append(data)

    cubes_hr, lams_hr, ra_hr, dec_hr = interpolate_data(ras, decs, lams, datas, method='linear')

    cubes, lams_seg, _nums = check_spatial_misalignment(cubes_hr, lams_hr)
    cube_final, lam_final = perform_linear_average(cubes, lams_seg)

    out_path = os.path.join(args.output_dir, args.output_name)

    # Write cube then reopen to inject header/WCS.
    hdu_out = fits.PrimaryHDU(data=cube_final)
    hdu_out.writeto(out_path, overwrite=True)
    hdu_out = fits.open(out_path)[0]

    # Construct a WCS for the new grid.
    hdu_out.header = update_header_with_wcs(hdu_out.header, ra_hr, dec_hr, lam_final, ra_offset=7)
    hdu_out.writeto(out_path, overwrite=True)

    print('Wrote uniform cube to:', out_path)


if __name__ == '__main__':
    main()
