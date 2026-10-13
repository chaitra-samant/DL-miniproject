# Cue-Aware Pedestrian Crossing Intent and Trajectory Prediction with Explainable Surprise Mining

Work in progress. See `configs/jaad.yaml` for the evaluation protocol and its sources.

## Workflow
- Local machine (no GPU): code, configs, report. Kaggle CLI in `.venv`.
- Kaggle kernels (pushed from `kaggle/`):
  - `ped-intent-jaad-download`: official JAAD clips + annotations, used by later kernels as `kernel_sources`.
  - Stage A feature extraction (GPU, resumable), Stage B training (cached features only), demo rendering.

## Data
- JAAD: clips from the YorkU server, annotations from github.com/ykotseruba/JAAD (branch JAAD_2.0).
- Benchmarks use ground-truth boxes and track IDs. Our YOLO + ByteTrack pipeline is used only for the end-to-end demo.
