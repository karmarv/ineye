import datetime

import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def frame_overlay(frame, detections):
    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(data[5])
        # if the confidence is greater than the minimum confidence, draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        cv.putText(frame, str(class_id), (xmin + 5, ymin - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


def detect(frame, model_type="yolo", do_overlay=False):
    if model_type == "haar":
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, 1.3, 5)
    else: 
        # run the YOLO model on the frame
        detections = model(frame)[0]
    if do_overlay:
        frame_overlay(frame, detections)
    return detections