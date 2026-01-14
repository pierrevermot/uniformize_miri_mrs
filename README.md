# uniformize_miri_mrs

Small self-contained toolkit to:

1. Unfringe JWST/MIRI MRS *s3d* spectral cubes.
2. Resample 4 channel cubes onto a common grid and merge into one uniform cube.

## Expected input layout

Put your observation cubes in:

- `uniformize_miri_mrs/data/`

The scripts look for filenames containing `ch1`, `ch2`, `ch3`, `ch4`.

## Usage

### 1) Unfringe

```bash
python script_unfringe.py --data-dir ./data --pattern '*.fits' --overwrite
```

Outputs are written next to the inputs with `_unfringed.fits` appended.

### 2) Uniformize

```bash
python script_uniformize.py --data-dir ./data --pattern '*unfringed.fits' --output-dir ./output
```

## Dependencies

- `numpy`, `scipy`, `astropy`, `joblib`
- `jwst` (for `jwst.residual_fringe`)

## Notes

- The merge recipe in `perform_linear_average` is inherited from a specific
  project and assumes the segmentation returns 10 pieces.
  If your dataset differs, that function must be generalized.
