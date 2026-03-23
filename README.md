# Pickleball Analysis Pipeline

Professional broadcast-quality pickleball analysis using TrackNetV3 and YOLOv8.

## Features
- **Fused Ball Detection**: Combines TrackNetV3 heatmap, YOLOv8 bounding boxes, and HSV/Motion fallback.
- **Player Tracking**: Consistent ID assignment for up to 4 players.
- **Shot Classification**: DINK, DRIVE, LOB, SMASH, DROP classification.
- **Broadcast Overlay**: High-quality HUD with trajectory trails, player circles, and shot labels.
- **Analytics**: Auto-generated shot distribution charts and court heatmaps.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure Roboflow (optional for training/data merge):
   ```bash
   export ROBOFLOW_API_KEY=your_key_here
   ```

3. Configure paths in `config.py` or use CLI arguments.

## Usage

### Run Pipeline
```bash
python3 pipeline.py --input video.mp4 --output tracked.mp4 --yolo-weights best.pt --tracknet-weights tracknet_best.pt
```

### Data Preparation
```bash
python3 data/download_datasets.py
python3 data/merge_datasets.py
```

### Training
```bash
python3 train.py --epochs 80
```

## Credits
Based on the Pickleball CV pipeline.
