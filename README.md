# Quantifying the NTN Opportunity Gap: A Cross-Generation Railway Measurement Study

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24044888.svg)](https://doi.org/10.5281/zenodo.24044888)

Processing pipeline and derived railway deficiency maps for the CCNC 2027 paper.

The study analyses 36 million multi-operator radio measurements collected on
Great Britain mainline railways across 2G, 4G and 5G, classifying each serving
sample as coverage-limited, interference-limited or acceptable, and quantifies
how much of the railway deficiency a non-terrestrial network could realistically
address.

## Contents

| | |
|---|---|
| `*.py` | pipeline reproducing the figures and tables of the paper |
| `maps/` | derived railway deficiency maps, per cell and per generation |
| `data/uk_outline.geojson` | UK boundary, used as a backdrop for the map figure |

The raw measurements are not redistributed here.

## Data

The dataset is published by Network Rail on the Rail Data Marketplace and is
subject to its terms:

https://raildata.org.uk/dataProduct/P-8e7dbe99-011d-431e-85ad-06efc77217fc/overview

Download `Global_View_2G.csv`, `Global_View_4G.csv` and `Global_View_5G_1.csv`
into `data/`.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python build_serving.py          # cleaning and serving-cell reduction
.venv/bin/python landscape.py              # coverage and quality landscape
.venv/bin/python regimes.py                # regime classification and sweep
.venv/bin/python density3.py               # joint power and quality density
.venv/bin/python atlas.py                  # interference-limited geography
.venv/bin/python blackspots.py             # coverage budget
.venv/bin/python operator_coverage.py      # per-operator and best-of-N coverage
.venv/bin/python mobility.py               # coverage holes and persistence
.venv/bin/python failures.py               # outage-event spans
.venv/bin/python make_deficiency_maps.py   # derived maps
```

`build_serving.py` runs first and produces the caches the other scripts read.
Figures and tables are written to `results/`.

## Deficiency maps

`maps/deficiency_map.csv` gives one row per generation and per cell of about
2 km, with per-generation GeoJSON versions for GIS tools. Each row reports the
share of samples that are coverage-limited, interference-limited and
acceptable, the received power of the best available operator, and how
consistently the cell is in outage across repeated journeys. Thresholds follow
the paper: RxLev and C/I for 2G, RSRP and SINR for 4G and 5G.

Shares in the maps are per cell. Weighting them by `n_samples` reproduces the
sample-weighted figures reported in the paper.

## Environment

Python 3.12 with numpy, pandas, pyarrow and matplotlib; see `requirements.txt`.
No geospatial toolkit is required.

## Licence

The code is released under the MIT licence (see `LICENSE`). The underlying
measurements are not covered by it: they remain subject to the Rail Data
Marketplace terms. The derived maps are aggregated products computed from that
dataset; preserve this attribution if you redistribute them.

`data/uk_outline.geojson` is third-party open data, included only as a figure
backdrop.

## Citation

```bibtex
@inproceedings{veisi2027ntngap,
  title     = {Quantifying the {NTN} Opportunity Gap: A Cross-Generation
               Railway Measurement Study},
  author    = {Veisi, Farzad and Choda, David and Secci, Stefano},
  booktitle = {Proc. IEEE Consumer Communications and Networking Conference
               (CCNC)},
  year      = {2027}
}
```
