import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

import supervision as sv

# define model path
model_name = "best.pt"
model_path = os.path.join(".", "weights", model_name)
device = "mps:0" if torch.backends.mps.is_available() else "cpu"
model = YOLO(model_path).to(device=device)
model.fuse()

# model config 
confidence_threshold = 0.7
iou_threshold = 0.5

# define video path 
video_name = "people_in_the_sea.mov"
video_path = os.path.join(".", "videos", video_name)
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
frame_w, frame_h = video_info.resolution_wh

ZONE_POLYGON = np.array([
    [0, frame_h],
    [frame_w, frame_h],
    [1550, 600],
    [350, 600]
])

# Initialize ByteTrack
byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence_threshold)

frame_generator = sv.get_video_frames_generator(video_path)

# annotator scale 
thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh
    )

text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh
    )

# define annotator instance
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=thickness
    )

label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.TOP_RIGHT,
)

# define polygon zone and annotator
zone = sv.PolygonZone(polygon=ZONE_POLYGON)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, 
    color=sv.Color.from_hex("#5ab23d"), 
    text_scale=(text_scale*2),
    thickness=thickness,
    text_thickness=thickness)

# Define the output video path and video sink
output_video_name = "output.mp4"
output_video_path = os.path.join(".", "videos", output_video_name)

with sv.VideoSink(output_video_path, video_info) as sink:
    for frame in frame_generator:
        result = model(source=frame)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections.with_nms(threshold=iou_threshold)

        # Perform tracking
        detections = byte_track.update_with_detections(detections=detections)

        # Zone annotation
        zone_count = zone.trigger(detections=detections)

        # Extract class names and track IDs for the detections
        labels = [
            f" {model.names[int(class_id)]} id:{tracker_id}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = frame.copy()

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
            )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
            )

        # display zone trigger
        safe_tourist = len(zone_count[zone_count == 1])
        annotated_frame = zone_annotator.annotate(scene=annotated_frame, label=f"Safe Zone : {safe_tourist}")

        # display all tourist
        cv2.putText(
            annotated_frame,
            f"All Tourist: {len(detections)}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 0, 0),
            5
        )

        # Write the frame to the video sink
        sink.write_frame(annotated_frame)

        cv2.imshow("annotated_frame", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()