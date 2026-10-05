# Deployment Checklist: Multi-Stage Mapping Pipeline

Use this checklist to ensure the pipeline is properly deployed and configured for your use case.

## Pre-Deployment

### System Requirements
- [ ] Python 3.8+ installed
- [ ] 8+ GB RAM available
- [ ] NVIDIA GPU with CUDA support (recommended) or CPU fallback
- [ ] ~500 MB free disk space (per 100 images)

### Installation
- [ ] Core dependencies installed: `pip install -r requirements.txt`
- [ ] Glue Factory installed: `pip install git+https://github.com/cvg/glue-factory.git`
- [ ] LoFTR installed: `pip install git+https://github.com/zju3dv/LoFTR.git`
- [ ] COLMAP installed: `brew install colmap` (or `apt-get install colmap`)
- [ ] Verify COLMAP: `colmap -h` returns version info

### Repository Structure
- [ ] New files exist in `mapping/pipeline/`:
  - [ ] `config.py`
  - [ ] `gps_pair_selector.py`
  - [ ] `feature_matcher.py`
  - [ ] `match_writer.py`
  - [ ] `colmap_mapper.py`
  - [ ] `pipeline.py`
  - [ ] `examples.py`
  - [ ] `integration_guide.py`
  - [ ] `ARCHITECTURE.md`
  - [ ] `IMPLEMENTATION_SUMMARY.md`
- [ ] `requirements.txt` updated with new dependencies

---

## Pre-Processing

### Image Preparation
- [ ] Images in single directory (e.g., `./input_images/`)
- [ ] Images in standard format (JPG/PNG)
- [ ] Image resolution: 1920×1440 or similar (adjust if >8GB VRAM)
- [ ] No corrupted/grayscale images in directory

### Metadata Preparation
- [ ] Per-image JSON metadata with GPS coordinates
  - [ ] Example format documented in `ARCHITECTURE.md`
  - [ ] One JSON per image (same filename stem)
  - [ ] Fields: `latitude`, `longitude`, `altitude`, `timestamp`
- [ ] OR single CSV with all metadata (to be implemented if needed)

### GPS Validation
- [ ] GPS coordinates in expected range (e.g., ~42.3°N, -71.0°W for Boston)
- [ ] No obvious GPS outliers (>1 km from other images)
- [ ] Timestamps or image ordering correct (temporal sequence)

---

## Configuration & Tuning

### Default Configuration
- [ ] Read `ARCHITECTURE.md` tuning guide (section "Tuning Guide")
- [ ] Defaults chosen for: balanced texture + grassy areas
  - [ ] `min_inliers = 120`
  - [ ] `min_keypoints = 800`
  - [ ] `max_reproj_error = 2.0 px`
  - [ ] `k_neighbors = 15`
  - [ ] `temporal_radius = 4`

### Scenario-Specific Tuning
Choose ONE of the below based on your imagery:

#### For Well-Textured Areas (Urban, Forest)
- [ ] Increase inlier threshold: `min_inliers = 200`
- [ ] Decrease temporal neighbors: `temporal_radius = 2`
- [ ] Stricter reprojection error: `max_reproj_error = 1.0`
- [ ] (Less LoFTR fallback expected)

#### For Grassy/Texture-Poor Areas (Agriculture, Open Field)
- [ ] Decrease inlier threshold: `min_inliers = 80`
- [ ] Increase temporal neighbors: `temporal_radius = 8`
- [ ] Looser reprojection error: `max_reproj_error = 3.0`
- [ ] (More LoFTR fallback expected)

#### For Speed-Critical Deployment (Embedded/Realtime)
- [ ] Enable EfficientLoFTR: `use_efficient_loftr = True`
- [ ] Use CPU: `device = "cpu"` (if no GPU)
- [ ] Reduce GPS pairs: `k_neighbors = 10`
- [ ] Reduce temporal neighbors: `temporal_radius = 2`

#### For Maximum Quality (Publication/Archive)
- [ ] Strict matching: `min_inliers = 200`, `max_reproj_error = 1.0`
- [ ] Maximize pairs: `k_neighbors = 25`, `temporal_radius = 6`
- [ ] GPU required: `device = "cuda"`
- [ ] Allow longer runtime (30+ minutes)

---

## Test Run

### Small-Scale Test
- [ ] Create test dataset: 10-20 images
- [ ] Run pipeline: `python -m mapping.pipeline.pipeline ./test_images ./test_output cuda`
- [ ] Expected runtime: 30-60 seconds
- [ ] Check output:
  - [ ] `features.h5` created (>1 MB)
  - [ ] `matches.h5` created (>100 KB)
  - [ ] `database.db` created (>1 MB)
  - [ ] COLMAP log appears (verbose output)

### Verify Results
- [ ] Results dictionary contains:
  - [ ] `status: "success"`
  - [ ] `pairs_matched > 0`
  - [ ] `verified_matches > 0`
  - [ ] `sp_lg_matches + loftr_fallbacks == pairs_matched`
- [ ] Point cloud generated (if COLMAP succeeded)
  - [ ] `colmap_model/sparse/0/model.ply` exists
  - [ ] File size > 1 MB (for 20 images)

### Quick Validation Commands
```bash
# Check h5 files
h5ls -A test_output/features.h5 | head -20
h5ls -A test_output/matches.h5 | head -20

# Check COLMAP DB
sqlite3 test_output/database.db "SELECT COUNT(*) FROM images;"
sqlite3 test_output/database.db "SELECT COUNT(*) FROM matches;"

# Visualize point cloud
open test_output/colmap_model/sparse/0/model.ply  # macOS
# or use CloudCompare / Meshlab on other platforms
```

---

## Performance Validation

### Matching Statistics
- [ ] Log sample output from test run:
  - [ ] # pairs selected
  - [ ] # pairs matched
  - [ ] Average inliers per pair
  - [ ] % using SuperPoint+LightGlue
  - [ ] % using LoFTR fallback

### Expected Ranges (for 20 typical aerial images)
| Metric | Expected Range | If Below | If Above |
|--------|----------------|----------|----------|
| Pairs selected | 15-50 | GPS too sparse | Increase k |
| Pairs matched | 10-40 | Check GPS/overlap | Normal |
| Avg inliers/pair | 50-300 | Grassy? Use LoFTR | Good signal |
| SP+LG %; LoFTR % | 60%/40% to 90%/10% | Depends on texture | Check |
| Total compute time | 1-10 min | Fast (good GPU) | Slow (CPU?) |

### SfM Reconstruction Quality
- [ ] COLMAP mapper completed successfully (not failed/crashed)
- [ ] Point cloud visible and reasonable
- [ ] Camera poses estimated (not singular)
- [ ] Bundle adjustment converged

---

## Production Deployment

### Code Integration
- [ ] Pipeline integrated into existing codebase (if needed)
  - [ ] Import paths correct
  - [ ] Config passed properly
  - [ ] Output paths handled
- [ ] Backwards compatibility checked (old code still works)

### Automation
- [ ] Create wrapper script for automated runs:
  ```bash
  #!/bin/bash
  python -m mapping.pipeline.pipeline \
    /path/to/images \
    /path/to/output \
    cuda
  ```
- [ ] Cron job / scheduler configured (if needed)
- [ ] Error handling implemented (check `results['status']`)
- [ ] Logging to file configured

### Monitoring
- [ ] Setup script to check output validity:
  ```python
  results = run_mapping_pipeline(...)
  assert results['status'] == 'success'
  assert results['verified_matches'] > 100
  ```
- [ ] Email alerts on failure (optional)
- [ ] Logging of statistics to database (optional)

### Resource Management
- [ ] GPU memory monitoring (use `nvidia-smi`)
  - [ ] No OOM (out of memory) errors
  - [ ] GPU utilization >70% during matching
- [ ] Disk space monitoring
  - [ ] Temp files cleaned up
  - [ ] Output archived/compressed if needed
- [ ] CPU/memory monitoring on CPU-only mode
  - [ ] No excessive swapping
  - [ ] Reasonable runtime (<1 hour for 100 images)

---

## Documentation

### For Your Team
- [ ] README updated with new pipeline reference
- [ ] Config examples documented with comments
- [ ] Output format documented (features.h5, matches.h5, COLMAP DB)
- [ ] Troubleshooting guide created (copy from `ARCHITECTURE.md`)

### For Future Reference
- [ ] Paper citations added (`IMPLEMENTATION_SUMMARY.md`)
- [ ] Benchmark results recorded (runtime, GPU memory, quality)
- [ ] Known issues logged (if any)
- [ ] Lessons learned documented

---

## Post-Deployment Monitoring

### First Week
- [ ] Run on 5-10 real datasets
- [ ] Document actual runtimes vs estimates
- [ ] Verify point cloud quality with domain experts
- [ ] Check for any edge cases / failures

### Ongoing
- [ ] Monthly check: "Did pipeline work for latest images?"
- [ ] Quarterly: Re-tune thresholds if needed
- [ ] Yearly: Review for updates/improvements

### Performance Tracking
- [ ] Keep log of:
  - [ ] Date run
  - [ ] # images
  - [ ] Matching statistics
  - [ ] Runtime
  - [ ] Point cloud quality (subjective: poor/ok/good)
  - [ ] Any issues encountered

---

## Rollback Plan

If issues arise:

1. **Quick Rollback** (revert to old stitcher)
   - [ ] Old stitcher code still available in git history
   - [ ] Restore: `git checkout HEAD~N -- stitcher.py`

2. **Investigate Issues**
   - [ ] Review pipeline output (verbose logs)
   - [ ] Check GPS data validity
   - [ ] Review config settings
   - [ ] Check input image quality

3. **Refinement**
   - [ ] Adjust config (try different tuning)
   - [ ] Re-run on same dataset
   - [ ] Compare results (point cloud, stats)

4. **Fix & Redeploy**
   - [ ] Once resolved, commit changes
   - [ ] Tag as stable release
   - [ ] Notify team of fix

---

## Sign-Off

- [ ] **Tested by**: _______________  Date: __________
- [ ] **Approved by**: _______________  Date: __________
- [ ] **Deployed to Production**: _______________  Date: __________

---

## Contact & Support

For issues or questions:
1. Check `ARCHITECTURE.md` tuning guide
2. Review `IMPLEMENTATION_SUMMARY.md` troubleshooting
3. Run examples: `python mapping/pipeline/examples.py --example 1`
4. Review git history: `git log --oneline mapping/pipeline/`
5. Contact: [Team lead / maintainer]

