# Computer-vision---Speed-detection
Traffic Speed Detection

Simple highway vehicle speed estimation using YOLOv11n and OpenCV. Vehicles are detected, tracked, and their speed (km/h) is estimated; boxes turn red when speed > limit.

### Installation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Usage


### Set or redo ROIs interactively (two directions):
```
python traffic_speed_limit.py --video traffic.mp4 --setup 2 --config config.yaml
```
### Run on a video:
```
python traffic_speed_limit.py --video traffic.mp4 --config config.yaml
```


## Output

Processed video is saved to:

out/output.mp4

## Files

> traffic_speed_limit.py — main script

> config.yaml — parameters (speed limit, ROIs, thresholds)

> traffic.mp4 — input video

> yolo11n.pt — YOLOv11 Nano weights

> out/ — results folder

## Model

YOLOv11n (yolo11n.pt) for object detection; ByteTrack for multi-object tracking (via Ultralytics). Works fully offline.
