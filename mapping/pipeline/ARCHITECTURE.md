# Multi-Stage Mapping Pipeline (Architecture B)

A production-ready framework for drone mapping with **GPS-aware pair selection** and **hybrid feature matching** (SuperPoint + LightGlue → LoFTR cascade) optimized for grassy/texture-poor areas.

## Overview

This pipeline implements a **cascade/fallback strategy** for robust image matching in challenging environments:

```
GPS Coordinates
       ↓
[GPS k-NN + Temporal neighbors] → Candidate image pairs
       ↓
For each pair:
  ├─ Stage 1: SuperPoint + LightGlue (fast)
  │  └─ RANSAC verification (Essential/Fundamental matrix)
  │  └─ If low inliers (<120) or few keypoints (<800) → fallback
  │
  └─ Stage 2: LoFTR (semi-dense, detector-free)
     └─ RANSAC verification
     └─ Accept inliers
       ↓
[Verified matches] → COLMAP SfM
       ↓
[Point cloud + aligned model]
```

## Architecture

### Module Structure

```
mapping/pipeline/
├── config.py                  # Configuration dataclasses with defaults
├── gps_pair_selector.py      # GPS k-NN + temporal pair selection
├── feature_matcher.py        # SuperPoint+LightGlue/LoFTR cascade matcher
├── match_writer.py           # Export to h5 (hloc) + COLMAP database
├── colmap_mapper.py          # COLMAP integration (mapper, BA, alignment)
└── pipeline.py               # Main orchestration (load → select → match → SfM)
```

### Key Components

#### 1. **GPS Pair Selector** (`gps_pair_selector.py`)
- **GPS k-NN**: Selects `k` nearest neighbors per image (default k=15)
- **Temporal neighbors**: Includes ±4 consecutive frames (helps with GPS jitter)
- Outputs: List of candidate (image_i, image_j) pairs for matching

**Config**:
```python
GPSPairSelectorConfig:
  k_neighbors: int = 15
  temporal_neighbors_radius: int = 4
  gps_max_distance_deg: float = 0.01  # ~1 km at equator
  use_temporal_neighbors: bool = True
```

#### 2. **Multi-Stage Feature Matcher** (`feature_matcher.py`)
Implements cascade matching with three main classes:

##### a) **SuperPointLightGlueMatcher**
- Uses Glue Factory (Apache-2.0 licensed)
- Fast on well-textured/well-overlapped regions
- Output: sparse keypoints + descriptors → matches

##### b) **RANSACVerifier**
- Geometric verification via Essential or Fundamental matrix
- Inlier/outlier filtering
- Reprojection error computation
- **Default thresholds**:
  - `max_reproj_error_px`: 2.0 pixels
  - `confidence`: 0.999
  - `max_iters`: 10,000

##### c) **LoFTRMatcher**
- Detector-free semi-dense matching
- Used as fallback when SP+LG fails
- Handles grassy/texture-poor areas well

##### d) **MultiStageFeatureMatcher** (orchestrator)
```python
MultiStageFeatureMatcher(
  min_inliers=120,           # Fallback if fewer inliers
  min_keypoints=800,         # Fallback if fewer keypoints per image
  max_reproj_error_px=2.0,   # Acceptance threshold
  use_efficient_loftr=False  # Set to True for faster LoFTR
)
```

**Decision tree**:
```
SP+LG extraction
  ├─ keypts1 < 800 OR keypts2 < 800?
  │  └─ YES → use LoFTR
  │
  └─ NO → match with LightGlue
     └─ < 8 matches?
     │  └─ YES → use LoFTR
     │
     └─ NO → RANSAC verification
        └─ inliers < 120 OR reproj_error > 2.0 px?
           └─ YES → use LoFTR
           └─ NO → accept (success)
```

#### 3. **Match Writer** (`match_writer.py`)
Exports verified matches in two formats:

##### a) **h5 (hloc format)**
```
features.h5
├── image1.jpg
│   ├── keypoints (N, 2)
│   └── descriptors (N, 256)
├── image2.jpg
│   ├── keypoints (M, 2)
│   └── descriptors (M, 256)
...

matches.h5
├── image1.jpg_image2.jpg → (P, 2) match indices
├── image1.jpg_image3.jpg → (Q, 2) match indices
...
```

**Advantages**: Reusable for multiple SfM tools, easy to debug/visualize

##### b) **COLMAP Database**
SQLite with schema:
- `cameras`: Camera calibration
- `images`: Image metadata + pose priors
- `keypoints`: Image keypoint locations
- `descriptors`: Feature descriptors
- `matches`: Verified match pairs
- `two_view_geometries`: Epipolar geometry (E/F matrix)

#### 4. **COLMAP Mapper** (`colmap_mapper.py`)
Wrapper around COLMAP binaries:
- **Mapper**: Incremental SfM (track creation, pose estimation)
- **Bundle Adjuster**: Non-linear refinement
- **Model Aligner**: GPS-to-local coordinate alignment
- **Exporter**: Point cloud export (PLY, OBJ)

**Config**:
```python
COLMAPMapperConfig:
  mapper_bin: str = "colmap"
  ba_refine_principal_point: bool = True
  ba_refine_extra_params: bool = True
  min_track_length: int = 2
  min_num_matches: int = 120
```

#### 5. **Pipeline Orchestrator** (`pipeline.py`)

Main entry point: `MultiStageMappingPipeline`

**Workflow**:
1. Load images from directory + parse per-image JSON metadata (GPS, altitude, yaw)
2. Select candidate pairs (GPS k-NN + temporal)
3. Match each pair (SuperPoint+LightGlue → LoFTR cascade)
4. Export verified matches to h5 + COLMAP DB
5. Run COLMAP incremental SfM (if installed)
6. Optionally align to GPS + export point cloud

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for speed)

### Step 1: Core dependencies
```bash
cd mapping/pipeline
pip install -r requirements.txt
```

### Step 2: Glue Factory (SuperPoint + LightGlue)
```bash
pip install git+https://github.com/cvg/glue-factory.git
# or with conda:
# conda install -c conda-forge glue-factory
```

### Step 3: LoFTR
```bash
pip install git+https://github.com/zju3dv/LoFTR.git
```

### Step 4: COLMAP
```bash
# macOS
brew install colmap

# Ubuntu
sudo apt-get install colmap

# Or build from source: https://colmap.github.io/install.html
```

Verify COLMAP:
```bash
colmap -h
```

---

## Usage

### Basic Usage

```python
from mapping.pipeline.pipeline import run_mapping_pipeline

results = run_mapping_pipeline(
    image_dir="/path/to/images",
    output_dir="/path/to/output",
    device="cuda"  # or "cpu"
)

print(f"Matched pairs: {results['pairs_matched']}")
print(f"Total inliers: {results['verified_matches']}")
print(f"Output h5: {results['features_h5']}")
print(f"Output COLMAP DB: {results['colmap_db']}")
print(f"Point cloud: {results['sfm_results']['point_cloud_path']}")
```

### Command line

```bash
python -m mapping.pipeline.pipeline /path/to/images /path/to/output cuda
```

### Custom Configuration

```python
from mapping.pipeline.config import PipelineConfig, MatcherConfig, GPSPairSelectorConfig

config = PipelineConfig()

# Adjust matching thresholds
config.matcher.min_inliers_sp_lg = 150  # stricter
config.matcher.max_reprojection_error_px = 1.5  # stricter

# Adjust GPS/temporal selection
config.gps_selector.k_neighbors = 20
config.gps_selector.temporal_neighbors_radius = 2

results = run_mapping_pipeline(
    image_dir="/path/to/images",
    output_dir="/path/to/output",
    device="cuda",
    config=config
)
```

---

## Expected Image Metadata Format

The pipeline expects per-image JSON metadata (same filename stem as image):

```json
{
  "image": {
    "telemetry": {
      "timestamp": 1234567890.5,
      "gps": {
        "latitude": 42.3601,
        "longitude": -71.0589,
        "altitude": 120.5
      },
      "planeYaw": 45.0,
      "camera_model": "PINHOLE",
      "calibration": {
        "fx": 1920.0,
        "fy": 1920.0,
        "cx": 1920.0,
        "cy": 1440.0
      }
    }
  }
}
```

If no JSON found, dummy metadata (0,0) GPS is used.

---

## Output Structure

```
output_dir/
├── features.h5                      # Keypoints + descriptors (hloc format)
├── matches.h5                       # Verified matches (hloc format)
├── database.db                      # COLMAP database (SQLite)
└── colmap_model/
    ├── sparse/
    │   └── 0/
    │       ├── cameras.txt
    │       ├── images.txt
    │       ├── points3D.txt
    │       └── model.ply             # Point cloud
    └── logs/
        └── mapper.log
```

---

## Tuning Guide

### For Grassy Fields (Low Texture)

**Recommendations**:
1. **Lower inlier threshold** → enables LoFTR fallback more aggressively
   ```python
   config.matcher.min_inliers_sp_lg = 80  # was 120
   ```

2. **Increase temporal neighbors** → more frame pairs (compensate for GPS jitter)
   ```python
   config.gps_selector.temporal_neighbors_radius = 8
   ```

3. **Looser reprojection error** → account for motion blur
   ```python
   config.matcher.max_reprojection_error_px = 3.0  # was 2.0
   ```

### For High-Overlap, Well-Textured Areas

**Recommendations**:
1. **Increase inlier threshold** → be selective
   ```python
   config.matcher.min_inliers_sp_lg = 200
   ```

2. **Reduce temporal neighbors** → save compute
   ```python
   config.gps_selector.temporal_neighbors_radius = 2
   ```

3. **Stricter reprojection error**
   ```python
   config.matcher.max_reprojection_error_px = 1.0
   ```

### GPU vs CPU

- **CUDA (GPU)**: ~50-100× faster, but requires NVIDIA GPU
  ```python
  results = run_mapping_pipeline(..., device="cuda")
  ```
  
- **CPU**: Slower, but works everywhere
  ```python
  results = run_mapping_pipeline(..., device="cpu")
  ```

### Using EfficientLoFTR (if LoFTR too slow)

```python
config.matcher.use_efficient_loftr = True
```

(Requires separate installation: `pip install git+https://github.com/zju3dv/EfficientLoFTR.git`)

---

## Debugging & Visualization

### Check matching statistics
```python
results = run_mapping_pipeline(...)
print(f"SuperPoint+LightGlue: {results['sp_lg_matches']} pairs")
print(f"LoFTR fallbacks: {results['loftr_fallbacks']} pairs")
print(f"Avg inliers/pair: {results['verified_matches'] / results['pairs_matched']:.1f}")
```

### Read h5 matches
```python
import h5py

with h5py.File("features.h5") as f:
    for img_name in f.keys():
        kpts = f[img_name]["keypoints"][:]
        print(f"{img_name}: {len(kpts)} keypoints")

with h5py.File("matches.h5") as f:
    for pair_key in f.keys():
        matches = f[pair_key][:]
        print(f"{pair_key}: {len(matches)} matches")
```

### Inspect COLMAP database
```bash
sqlite3 database.db
sqlite> SELECT COUNT(*) FROM images;
sqlite> SELECT COUNT(*) FROM keypoints;
sqlite> SELECT COUNT(*) FROM matches;
```

---

## Paper References

- **SuperPoint** (DeTone et al., 2018): https://arxiv.org/abs/1712.07629
- **LightGlue** (Lindenberger et al., 2023): https://arxiv.org/abs/2306.13643
- **LoFTR** (Sun et al., 2021): https://arxiv.org/abs/2104.00680
- **COLMAP** (Schönberger & Frahm, 2016): https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-from-Motion_Revisited_CVPR_2016_paper.pdf

---

## License

- **SuperPoint + LightGlue** (Glue Factory): Apache-2.0
- **LoFTR**: CC-BY-NC (check their repo for updates)
- **COLMAP**: BSD 3-clause

---

## Troubleshooting

### "No matches found"
- Increase temporal neighbors: `temporal_neighbors_radius = 8`
- Lower inlier threshold: `min_inliers_sp_lg = 80`
- Check GPS coordinates: are pairs actually overlapping?

### "COLMAP not found"
- Ensure COLMAP is installed and in PATH: `which colmap`
- Or pass explicit path: `COLMAPMapper(colmap_bin="/usr/local/bin/colmap")`

### Out of memory
- Reduce image resolution before pipeline (batch resize in preprocessing)
- Use CPU instead of GPU: `device="cpu"`
- Disable descriptor storage in COLMAP: `config.colmap_mapper.ba_refine_extra_params = False`

### Slow performance
- Enable GPU: `device="cuda"`
- Use EfficientLoFTR: `config.matcher.use_efficient_loftr = True`
- Reduce temporal neighbors (fewer pairs to match)

---

## Future Enhancements

- [ ] Texture-based pair routing (pre-check signal for grassy areas)
- [ ] Union/ensemble matching (combine SP+LG + LoFTR inliers)
- [ ] Parallelized pair matching (multi-GPU)
- [ ] hloc integration for seamless swapping of matchers
- [ ] Dense reconstruction (MVS after SfM)
- [ ] Real-time preview + interactive pair selection UI

