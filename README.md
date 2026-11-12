# Fault prediction in electrical distribution systems

Machine learning module of the paper *A Data-Driven Framework for Fault Prediction and Spatial
Prioritization in Electrical Distribution Systems* (under review, International Journal of
Electrical Power & Energy Systems).

The script compares 7 classifiers under 5 resampling strategies (repeated stratified 5-fold CV,
Friedman and Wilcoxon tests), tunes XGBoost with randomized search, explains it with SHAP,
calibrates its probabilities with isotonic regression and exports the predicted failure points
used by the GIS-MCDM stage. The GIS-MCDM, SMAA and discrete-event simulation stages are not
included here.

## Data

The fault records belong to the distribution utility and are not public. `datos_sinteticos.py`
generates a synthetic file with the same columns so the pipeline can be run end to end; its
results do not match the paper. Researchers interested in the original data may contact the
corresponding author.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv run datos_sinteticos.py                          # writes datasets/sintetico.csv
uv run prediccion_fallas.py datasets/sintetico.csv  # figures and tables in resultados/
```

With the original data in `datasets/tabla.csv`, `uv run prediccion_fallas.py` reproduces the
results reported in the paper.

## License

MIT
