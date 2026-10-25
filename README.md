Himalayan Landslide Susceptibility Modeling
Project overview
This project explores machine-learning methods for identifying
landslide-susceptibility patterns in the Himalayan region, focusing on
Uttarakhand and nearby Himachal Pradesh. The intended product is a
mobile-friendly dashboard where a user enters a location and views an
estimated susceptibility result. Any short-term warning output is
experimental and is not an operational warning.
Current susceptibility dataset
Dataset: data/himalayan_landslide_v1_dataset.csv
Columns: - LATITUDE - LONGITUDE - ELEVATION - SLOPE -
RAINFALL - LANDCOVER_CODE - LANDSLIDE --- target label (1 =
positive sample, 0 = background sample)
Dataset audit
The original CSV contains 23,018 rows, with 11,509 samples in each
class. An audit found 705 exact duplicate rows. Removing exact duplicate
rows produced a cleaned in-memory dataset of 22,313 rows:
  Item                                                           Count
  Original rows                                                 23,018
  Exact duplicate rows removed                                     705
  Cleaned rows                                                  22,313
  Positive samples (LANDSLIDE = 1)                            10,804
  Background samples (LANDSLIDE = 0)                          11,509
  Remaining exact duplicate rows                                     0
  Remaining repeated coordinates                                     0
  Coordinates with conflicting labels in the original audit          0
The cleaned dataset was created in notebook memory using
drop_duplicates(); this operation did not overwrite the original CSV.
Spatial structure and validation
Positive samples are spatially clustered: - Median nearest-positive
distance: approximately 0.272 km. - 86.09% of positive samples have
another positive sample within 1 km. - 99.64% have another positive
sample within 5 km.
Using an approximate 10 km grid, the cleaned dataset occupied 1,166
blocks: 587 contained both classes, 11 contained positive samples only,
and 568 contained background samples only. Positive samples were more
concentrated in the grid than background samples.
A group-based spatial split was used for exploratory evaluation. The
test set contained 5,822 samples. A diagnostic subset retaining test
samples at least 5 km from a training sample contained 1,026 rows (608
background and 418 positive). At 10 km, only 24 rows remained, too few
for a stable headline evaluation.
The grid and distance calculations are approximate. This evaluation does
not guarantee geographic independence or establish performance in
completely unseen regions.
Exploratory model results
A Gradient Boosting model using ELEVATION, SLOPE, RAINFALL, and
LANDCOVER_CODE was evaluated on the spatial test set and the 5 km
buffered diagnostic subset.
  Metric               Full spatial test set (n 5 km buffered subset (n =
                                       = 5,822)                    1,026)
  Accuracy                               0.8063                    0.8138
  Precision                              0.7767                    0.7256
  Recall                                 0.8595                    0.8732
  F1-score                               0.8160                    0.7926
  ROC-AUC                                0.8866                    0.8887
  PR-AUC / average                       0.8695                    0.8209
  precision
These are exploratory results on a constructed dataset, not validated
real-world performance. The dataset is close to balanced, so its
positive rate does not represent real-world landslide prevalence. Model
outputs should not be described as calibrated probabilities without
further calibration and validation.
Data provenance: known limitations
The original code that created himalayan_landslide_v1_dataset.csv has
not yet been located. The project README was empty, and the project
directory was not a Git repository during the audit.
The following details remain unverified and should be recovered from
source files, provider documentation, or project records before
publication: - Exact source inventory and inclusion criteria for
positive landslide samples. - How background samples were selected and
whether they are confirmed non-landslide locations. - Source, temporal
coverage, units, and extraction method for RAINFALL. - Source and
processing steps for ELEVATION and SLOPE. - Source, version, and
class mapping for LANDCOVER_CODE. - Sampling dates, geographic
coverage, and filters applied before the CSV was created.
Do not assume background samples are true negatives just because no
landslide is recorded at those locations. An incomplete inventory can
leave unrecorded landslides among background samples.
Important interpretation limits
1. Sampling design: The dataset has approximately balanced classes.
   Metrics and the test-set positive rate should not be interpreted as
   real-world event probability or alert precision.
2. Spatial clustering: Positive points are much more clustered than
   background points. Multiple points may represent nearby or related
   records; event-level independence has not been established.
3. Spatial validation: Grid-based splitting and distance-buffer
   checks reduce some proximity concerns but do not prove independent
   regional generalization.
4. Provenance gaps: Original feature sources and sampling
   procedures still need verification.
5. Operational use: This is a research prototype, not an official
   landslide warning system or a substitute for guidance from
   responsible authorities.
Project structure
Known project folders and notebooks include: - data/ --- datasets and
derived files. - notebooks/01_data_analysis.ipynb --- initial
susceptibility dataset analysis. -
notebooks/01_susceptibility_modeling.ipynb --- susceptibility data
audit and modeling. - notebooks/02_event_data.ipynb and
notebooks/03_event_data.ipynb --- temporal/event-data experiments. -
notebooks/04_model_training.ipynb --- model-training experiments and
later audits.
Notebook contents and intermediate outputs may evolve. Verify current
files and execution state before reproducing results.
Recommended next steps
1. Recover and document the original data-construction procedure and
   source references.
2. Verify feature units, temporal coverage, and land-cover class
   meanings against the actual source products.
3. Determine whether clustered positive records can be linked to
   individual events.
4. Keep a fixed, documented validation design and avoid repeated tuning
   against the final test set.
5. Evaluate on geographically independent data when suitable data
   become available.
6. Treat short-term prediction as a separate experimental task
   requiring time-aware labels and independent temporal validation.

## Verified event-rainfall provenance

The file `data/himalayan_event_rainfall.csv` contains 377 rows and 375 unique coordinate-date keys. Comparison with the NASA regional landslide inventory (`HMA_LS_Cat_point_v02.0.zip`) found 375 exact coordinate-date matches, using latitude and longitude rounded to four decimal places. This establishes a strong match between the event-rainfall records and the NASA inventory records used in the comparison.

This result does not establish that the NASA inventory was used to create the separate `himalayan_landslide_v1_dataset.csv` susceptibility dataset. The V1 positive coordinates had no exact coordinate matches with the 374 unique event-rainfall coordinates at four decimal places. The original V1 construction procedure and source inventory remain unverified.
