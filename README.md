# Mapping

A toolkit for processing, stitching, and analyzing drone imagery with support for DJI metadata extraction and various stitching algorithms.

## Features

- **DJI Metadata Extraction**:
  - Automatic extraction of GPS coordinates from DJI drone images
  - Parsing of altitude, heading, and timestamp data
  - Support for DJI's XMP metadata format
  - CSV export of extracted metadata
- **Image Stitching**:
  - OpenCV-based feature detection and matching using SIFT
  - Geolocation-aware stitching that considers image coordinates
  - Automatic center point tracking during stitching
- **Image Processing**:
  - Automatic rotation based on compass heading
  - Altitude-based image resizing
  - Coordinate-to-pixel mapping for accurate placement
- **Stitching Pipeline**:
  - Multi-pass stitching with overlap detection
  - Transparent blending of overlapping regions
  - Preservation of original center points
  - Automatic canvas sizing and image placement

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mapping.git
cd mapping
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### One-Command Pipeline (Recommended)

`stitch.sh` handles everything in a single command: it finds the Python environment, generates the DJI metadata CSV if one does not already exist, runs the full feature-based stitching pipeline, and writes the final map.

```bash
./stitch.sh <path_to_image_folder> [--no-alpha] [--output <file>]
```

**Examples:**
```bash
# Output saved to output.jpg (default, alpha blending on)
./stitch.sh /path/to/DJI_images

# Disable alpha blending
./stitch.sh /path/to/DJI_images --no-alpha

# Custom output filename
./stitch.sh /path/to/DJI_images --output my_map.jpg
```

**What it does, step by step:**
1. Locates Python — uses `.venv/bin/python` if present, otherwise falls back to `python3`/`python` on `$PATH`.
2. Checks for `<folder>.csv` next to the image folder. If missing, runs `dji_dataextraction.py` to generate it from XMP metadata embedded in the DJI images.
3. Runs `with_stitch_main.py` — SIFT feature matching with GPS-assisted placement, global refinement, harsh-break removal, and alpha-blended assembly.
4. Writes the final composite map to the specified output file.

> **Note:** Make the script executable once after cloning: `chmod +x stitch.sh`

---

### Manual Steps (Advanced)

### 1. Extract Metadata
First, extract the GPS and orientation metadata from your DJI images.
**Note**: The output CSV must be named `<folder_name>.csv` (e.g., if folder is `my_data`, csv must be `my_data.csv`) and placed in the same parent directory or the path expected by the main script.

```bash
python dji_dataextraction.py <path_to_image_folder> <path_to_image_folder>.csv
```

### 2. Generate Map
There are two modes available:

**Option A: Feature-Based Stitching (Recommended)**
Uses SIFT features to stitch images with high accuracy, using GPS for initial alignment.
```bash
python with_stitch_main.py <path_to_image_folder>
```

**Option B: GPS-Only Mosaic**
Places images solely based on their GPS coordinates. Faster but less accurate if GPS has drift.
```bash
python main.py <path_to_image_folder>
```


## Output

The toolkit generates:
- Stitched image output (`output.jpg`)
- Metadata CSV files containing:
  - Image filenames
  - GPS coordinates (latitude/longitude)
  - Altitude information
  - Flight direction (degrees clockwise from North)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pillow
- Additional dependencies listed in `requirements.txt`