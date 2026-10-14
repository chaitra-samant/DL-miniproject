| model               | ccd_source AP   | ccd_source mTTA   | ccd_source TTA_R80   | dota_source AP   | dota_source mTTA   | dota_source TTA_R80   |
|:--------------------|:----------------|:------------------|:---------------------|:-----------------|:-------------------|:----------------------|
| frame_lstm          | 1.000 ± 0.000   | 4.67 ± 0.05       | 4.37 ± 0.08          | 0.722 ± 0.016    | 3.59 ± 0.06        | 4.42 ± 0.14           |
| det_lstm            | 0.996 ± 0.006   | 4.63 ± 0.02       | 4.32 ± 0.04          | 0.800 ± 0.014    | 3.47 ± 0.14        | 4.10 ± 0.11           |
| det_graph           | 0.999 ± 0.000   | 4.62 ± 0.13       | 4.25 ± 0.20          | 0.814 ± 0.017    | 3.62 ± 0.15        | 4.00 ± 0.05           |
| main                | 1.000 ± 0.000   | 4.61 ± 0.08       | 4.21 ± 0.16          | 0.821 ± 0.012    | 3.47 ± 0.09        | 4.02 ± 0.09           |
| main_no_interaction | 1.000 ± 0.000   | 4.68 ± 0.01       | 4.43 ± 0.07          | not run          | not run            | not run               |
| main_no_app         | 1.000 ± 0.000   | 4.52 ± 0.08       | 4.06 ± 0.18          | not run          | not run            | not run               |
| main_no_geo         | 0.999 ± 0.000   | 4.61 ± 0.10       | 4.21 ± 0.17          | not run          | not run            | not run               |
| main_no_frame       | 0.989 ± 0.001   | 4.63 ± 0.03       | 4.01 ± 0.00          | not run          | not run            | not run               |
| main_gru            | 1.000 ± 0.000   | 4.63 ± 0.00       | 4.30 ± 0.00          | not run          | not run            | not run               |
| geo_only            | not run         | not run           | not run              | not run          | not run            | not run               |
| geo_seg             | not run         | not run           | not run              | not run          | not run            | not run               |