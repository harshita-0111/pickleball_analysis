# Pickleball Analysis Pipeline — Walkthrough

I have successfully converted your monolithic Colab notebook into a professional, modular Python repository. The new structure makes the code easier to maintain, test, and run locally.

## New Repository Structure

The code is now organized into logical modules:

- **`config.py`**: A central configuration file for all constants, thresholds, colors, and paths.
- **`models/`**: Contains the `TrackNetV3` architecture.
- **`tracking/`**: Houses the core logic for ball and player detection, Kalman filtering, and rally status.
- **`analysis/`**: Includes shot classification and bounce detection.
- **`visualization/`**: Professional broadcast HUD drawing, Bézier trajectory trails, and analytics charts.
- **`pipeline.py`**: The main entry point to run the video analysis with a CLI interface.
- **`train.py`**: A dedicated script for training the YOLOv8 model.
- **`data/`**: Scripts for downloading and merging datasets from Roboflow.

## How to Run

1.  **Install Requirements**:
    ```bash
    cd pickleball-analysis
    pip install -r requirements.txt
    ```

2.  **Run the Analysis**:
    ```bash
    python pipeline.py --input path/to/video.mp4 --output output.mp4
    ```

## Key Improvements

- **Modularity**: Individual components (like the Kalman filter or the Shot Classifier) can now be tested and improved in isolation.
- **CLI Interface**: No need to edit code to change input/output paths or model weights.
- **Local Compatibility**: Replaced Colab-specific paths and dependencies with standard Python practices.
- **Clean Documentation**: Added a comprehensive `README.md` and `requirements.txt`.

## Verification Results

- [x] **File Structure**: Verified all 15+ files and directories exist.
- [x] **Import Test**: Successfully verified that all modules can be imported without errors in a Python environment.
- [x] **CLI Help**: Verified that `pipeline.py` and `train.py` have functional argument parsing.

Your new repository is located at:
`/Users/harshitaukande/.gemini/antigravity/scratch/pickleball-analysis/`
