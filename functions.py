#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""functions.py

Self-contained utilities to:

1) Unfringe JWST/MIRI MRS *s3d* cubes using the JWST pipeline residual-fringe fitter.
2) Resample multiple spectral cubes onto a common (uniform) spatial and spectral grid.
3) Merge channel cubes into a single cube on that uniform grid.


Assumptions / conventions
------------------------
- Input data live in `./data/` (relative to `uniformize_miri_mrs/`).
- Input cubes are JWST pipeline MIRI MRS s3d products.
- The spectral axis is axis-3 in the FITS cube.
- For uniformization we assume the inputs correspond to 4 MIRI MRS channels.

Notes
-----
- This code uses `jwst.residual_fringe` which must be installed in the Python
  environment.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, median_filter

try:
    from jwst.residual_fringe.utils import fit_residual_fringes_1d as rf1d
except Exception as exc:  # pragma: no cover
    rf1d = None
    _JWST_IMPORT_ERROR = exc
else:
    _JWST_IMPORT_ERROR = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------
# Fringe-removal (unfringing)
# -----------------------------

def process_pixel(flux: np.ndarray, wave: np.ndarray, lam_scs: Sequence[Sequence[float]], channel: int) -> np.ndarray:
    """Remove the residual fringe pattern from a single spectrum (one spatial pixel).

    This wraps the JWST pipeline `fit_residual_fringes_1d` routine. If the fitter
    errors (e.g. singular matrix), we slightly expand the spectral fitting window
    to try to recover.

    Parameters
    ----------
    flux
        1D flux array.
    wave
        1D wavelength array (same length as `flux`).
    lam_scs
        List of wavelength ranges (in same wavelength units as `wave`) defining
        segments to unfringe.
    channel
        MIRI MRS channel index (1..4).

    Returns
    -------
    numpy.ndarray
        New flux array with fringes corrected.
    """
    if rf1d is None:  # pragma: no cover
        raise ImportError(
            "jwst.residual_fringe is required for unfringing but could not be imported"
        ) from _JWST_IMPORT_ERROR

    new_flux = flux.copy()
    for k in range(len(lam_scs)):
        x0 = int(np.argmin(abs(wave - lam_scs[k][0])))
        x1 = int(np.argmin(abs(wave - lam_scs[k][1])))
        while True:
            try:
                flux_segment = flux[x0:x1]
                wave_segment = wave[x0:x1]
                if flux_segment.size > 0 and np.min(flux_segment) > 0:
                    flux_cor = rf1d(flux_segment, wave_segment, channel=channel)
                    new_flux[x0:x1] = flux_cor
            except Exception as e:  # pragma: no cover
                log.error(e)
                log.info(
                    "%s, %s, %s, %s, %s, modifying spectral range (slightly)",
                    k,
                    x0,
                    x1,
                    int(np.argmin(abs(wave - lam_scs[k][0]))),
                    int(np.argmin(abs(wave - lam_scs[k][1]))),
                )
                x0 = max(0, x0 - 1)
                x1 += 1
                continue
            break

    return new_flux


def unfringe_cube(hdu: fits.HDUList, lam_scs: Sequence[Sequence[float]], channel: int) -> np.ndarray:
    """Unfringe a full IFU cube by processing each spatial pixel spectrum in parallel."""
    data = np.nan_to_num(hdu[1].data)
    header = hdu[1].header

    # Build wavelength axis from FITS WCS keywords (original behavior).
    lam = (
        (np.arange(int(header["NAXIS3"])) - int(header["CRPIX3"]) + 1) * float(header["CDELT3"]) + float(header["CRVAL3"])
    )

    ny = len(data[0])
    nx = len(data[0, 0])

    results = Parallel(n_jobs=-1)(
        delayed(process_pixel)(data[:, i, j], lam, lam_scs, channel) for i in range(ny) for j in range(nx)
    )

    new_data = data.copy()
    for idx, result in enumerate(results):
        i = idx // nx
        j = idx % nx
        new_data[:, i, j] = result

    return new_data


# -----------------------------
# Cube IO helpers
# -----------------------------

def find_fits_files(directory: str, pattern: str) -> List[str]:
    """Find FITS files matching a glob pattern in a directory."""
    return sorted(glob.glob(os.path.join(directory, pattern)))


def open_and_sort_fits_files(filenames: Sequence[str]) -> Tuple[List[fits.PrimaryHDU], np.ndarray]:
    """Open FITS files and sort them by a characteristic wavelength.

    For MIRI MRS s3d products, the cube itself is generally in extension 1.
    Here we use CRVAL3 from ext=1 as a monotonic ordering proxy.
    """
    hdus = []
    wls = []
    for fn in filenames:
        hdul = fits.open(fn)
        hdu = hdul[1]
        hdus.append(hdu)
        wls.append(float(hdu.header.get("CRVAL3", 0.0)))

    order = np.argsort(wls)
    hdus_sorted = [hdus[i] for i in order]
    wls_sorted = np.array(wls)[order]
    return hdus_sorted, wls_sorted


def extract_header_info(hdu: fits.PrimaryHDU) -> dict:
    """Return a dict of header keywords/values to preserve."""
    # The original code preserved almost everything from the first cube header.
    return dict(hdu.header)


def update_header_with_wcs(header, ra, dec, lam, ra_offset: float = 7):
    """Update a FITS header with WCS keywords for a (lam, dec, ra) cube.

    This is a minimal WCS writer.
    It sets the linear WCS solution for the grid that `extract_data_from_hdu` uses.

    Parameters
    ----------
    header
        FITS header to update.
    ra, dec, lam
        1D axis arrays.
    ra_offset
        Historical offset used in the original pipeline.

    Returns
    -------
    astropy.io.fits.Header
    """
    # Axis sizes
    header["NAXIS"] = 3
    header["NAXIS1"] = len(ra)
    header["NAXIS2"] = len(dec)
    header["NAXIS3"] = len(lam)

    # CTYPEs (simple; users can refine if needed)
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CTYPE3"] = "WAVE"

    # Pixel reference positions
    header["CRPIX1"] = 1
    header["CRPIX2"] = 1
    header["CRPIX3"] = 1

    # Reference values
    header["CRVAL1"] = float(ra[0]) + ra_offset * float(ra[1] - ra[0]) if len(ra) > 1 else float(ra[0])
    header["CRVAL2"] = float(dec[0])
    header["CRVAL3"] = float(lam[0])

    # Increments
    header["CDELT1"] = float(ra[1] - ra[0]) if len(ra) > 1 else 1.0
    header["CDELT2"] = float(dec[1] - dec[0]) if len(dec) > 1 else 1.0
    header["CDELT3"] = float(lam[1] - lam[0]) if len(lam) > 1 else 1.0

    return header


# -----------------------------
# Uniform grid generation
# -----------------------------

def extract_data_from_hdu(hdu: fits.PrimaryHDU) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract RA, Dec, wavelength, and data from a FITS cube HDU.

    Returns
    -------
    ra, dec, lam, data
        `data` is returned as (lam, dec, ra).
    """
    header = hdu.header
    data = np.nan_to_num(hdu.data)

    ra = -(
        (np.arange(int(header["NAXIS1"])) - int(header["CRPIX1"]) + 1) * float(header["CDELT1"]) + float(header["CRVAL1"])
    )
    dec = (
        (np.arange(int(header["NAXIS2"])) - int(header["CRPIX2"]) + 1) * float(header["CDELT2"]) + float(header["CRVAL2"])
    )
    lam = (
        (np.arange(int(header["NAXIS3"])) - int(header["CRPIX3"]) + 1) * float(header["CDELT3"]) + float(header["CRVAL3"])
    )
    return ra, dec, lam, data


def interpolate_data(
    ras: Sequence[np.ndarray],
    decs: Sequence[np.ndarray],
    lams: Sequence[np.ndarray],
    datas: Sequence[np.ndarray],
    method: str = "linear",
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Interpolate multiple cubes onto a single shared (RA, Dec, λ) grid."""
    coordinates = [ras, decs, lams]

    deltas = []
    deltas_max = []
    mins = []
    maxs = []

    # Find global min/max values and sampling steps across all cubes.
    for coords in coordinates:
        deltas_t, mins_t, maxs_t = [], [], []
        for co in coords:
            mins_t.append(np.min(co))
            maxs_t.append(np.max(co))
            deltas_t.append(abs(co[1] - co[0]))
        deltas.append(np.min(deltas_t))
        deltas_max.append(np.max(deltas_t))
        mins.append(np.min(mins_t))
        maxs.append(np.max(maxs_t))

    # Define a common high-resolution grid.
    ra_hr = np.arange(maxs[0], mins[0] - deltas[0] * 0.09, -deltas[0])
    dec_hr = np.arange(mins[1], maxs[1] + deltas[1] * 0.09, deltas[1])
    lam_hr = np.arange(mins[2], maxs[2] + deltas[2] * 0.09, deltas[2])

    cubes_hr: List[np.ndarray] = []
    lams_hr: List[np.ndarray] = []

    for ra, dec, lam, data in zip(ras, decs, lams, datas):
        # Smooth individual images to a common spatial resolution before resampling.
        data_new = []
        for k in range(len(data)):
            im = data[k]
            im_new = gaussian_filter(
                im, (deltas_max[0] / abs(ra[0] - ra[1]), deltas_max[0] / abs(ra[0] - ra[1]))
            )
            data_new.append(im_new)

        data_new = np.array(data_new)
        interp_gauss = RegularGridInterpolator([lam, dec, ra[::-1]], data_new, bounds_error=False, method=method)

        # Restrict the interpolation wavelengths to the cube's native span.
        ra_hr_t = ra_hr.copy()
        dec_hr_t = dec_hr.copy()
        lam_hr_t = np.arange(
            lam_hr[int(np.argmin(abs(lam_hr - np.min(lam))))] + deltas[2],
            lam_hr[int(np.argmin(abs(lam_hr - np.max(lam))))],
            deltas[2],
        )

        mesh_hr = np.meshgrid(lam_hr_t, dec_hr_t, ra_hr_t[::-1], indexing="ij")
        points = np.array([mesh_hr[0].ravel(), mesh_hr[1].ravel(), mesh_hr[2].ravel()]).T
        cube_hr = interp_gauss(points).reshape(mesh_hr[0].shape)
        cubes_hr.append(cube_hr)
        lams_hr.append(lam_hr_t)

    return cubes_hr, lams_hr, ra_hr, dec_hr


def check_spatial_misalignment(cubes_hr: Sequence[np.ndarray], lams_hr: Sequence[np.ndarray]):
    """Split cubes into overlap / non-overlap segments.
    """
    cubes = []
    lams = []
    nums = []

    for k in range(len(cubes_hr)):
        if (k - 1) >= 0:
            lam_hr_p = lams_hr[k - 1]
        else:
            lam_hr_p = np.array([0])

        cube_hr_t = cubes_hr[k]
        lam_hr_t = lams_hr[k]

        if (k + 1) < len(cubes_hr):
            lam_hr_n = lams_hr[k + 1]
        else:
            lam_hr_n = np.array([1e10])

        i_0 = int(np.argmin(abs(lam_hr_t - np.max(lam_hr_p))))
        i_1 = int(np.argmin(abs(lam_hr_t - np.min(lam_hr_n))))

        cube_0 = cube_hr_t[: i_0 + 1]
        lam_0 = lam_hr_t[: i_0 + 1]
        cube_2 = cube_hr_t[i_1:]
        lam_2 = lam_hr_t[i_1:]

        if len(cube_2) > 1:
            cube_1 = cube_hr_t[i_0 + 1 : i_1]
            lam_1 = lam_hr_t[i_0 + 1 : i_1]
        else:
            cube_1 = cube_hr_t[i_0 + 1 :]
            lam_1 = lam_hr_t[i_0 + 1 :]

        if len(cube_0) <= 1:
            cube_1 = cube_hr_t[: i_1]
            lam_1 = lam_hr_t[: i_1]

        if len(cube_0) > 1:
            nums.append(0)
            cubes.append(cube_0)
            lams.append(lam_0)
        if len(cube_1) > 1:
            nums.append(1)
            cubes.append(cube_1)
            lams.append(lam_1)
        if len(cube_2) > 1:
            nums.append(2)
            cubes.append(cube_2)
            lams.append(lam_2)

    return cubes, lams, nums


def perform_linear_average(cubes: Sequence[np.ndarray], lams: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly blend overlap regions and concatenate non-overlapping parts.
    """

    def linear_interpolate(cube1, cube2):
        return cube1 * (1 - np.linspace(0, 1, len(cube1)))[:, None, None] + cube2 * np.linspace(0, 1, len(cube2))[:, None, None]

    cmid_a = linear_interpolate(cubes[1], cubes[2])
    cmid_b = linear_interpolate(cubes[4], cubes[5])
    cmid_c = linear_interpolate(cubes[7], cubes[8])
    cube_final = np.concatenate([cubes[0], cmid_a, cubes[3], cmid_b, cubes[6], cmid_c, cubes[9]])
    lam_final = np.concatenate([
        lams[0],
        np.mean(lams[1:3], 0),
        lams[3],
        np.mean(lams[4:6], 0),
        lams[6],
        np.mean(lams[7:9], 0),
        lams[9],
    ])

    return cube_final, lam_final


def split_continuum_emission(data: np.ndarray, filter_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Split a cube into a median-filter continuum and a residual emission-line cube."""
    continuum = median_filter(data, size=(filter_size, 1, 1))
    emission_line_cube = data - continuum
    return continuum, emission_line_cube
