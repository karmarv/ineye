# INEYE
Vision data exploration and frame sampling toolkit for videos
- Visualize video streams and run basic detections
  ``` python viz.py --video data/VIRAT_S_050201_05_000890_000944.mp4 ```
- Sample frames at fixed frequency (save-frames=0.1 for FPS/0.1 or 10xFPS rate) or based on detections 
  ``` python viz.py --output-path "./data/output" --save-frames 0.1 --video-ext "mp4" --video data/```
- Tool for video description
  - Exif Tool [https://exiftool.org/index.html]
  - FFMPEG [https://ffmpeg.org/]

##### Future Functions:
- Multi Camera Capture (10 FPS at an interval-period of 1 Second)
- Detect object on a configured part/zone of an image 
- Detect trajectory on the detected object
- Send and archive event notifications (email or ifttt)

