# Safe Tourist Detection and Tracking using YOLOv8 and ByteTrack with Supervision

This project focuses on implementing a system for safe tourist detection and tracking in videos using YOLOv8 for object detection and ByteTrack for high-precision object tracking, with additional supervision features.

## Overview

Tourist safety is paramount in crowded tourist spots, especially those near bodies of water like beaches. This project aims to address this concern by developing a system that can detect and track tourists in videos, ensuring their safety within predefined zones. The system utilizes state-of-the-art deep learning models for object detection and tracking, combined with supervision techniques to enhance safety measures.

## Features

- **Object Detection**: YOLOv8, a highly efficient object detection model, is employed to detect tourists in videos with high accuracy and speed. ([Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics))
- **Object Tracking**: ByteTrack, a robust object tracking algorithm, is used to track tourists over consecutive frames, ensuring precise localization.
- **Zone Supervision**: The system incorporates zone supervision, allowing users to define specific areas where tourists should be present. Any deviation from these zones triggers an alert, enhancing safety measures. ([Supervision Repo](https://github.com/roboflow/supervision))
- **Adjustable Thresholds**: Confidence and IoU thresholds can be adjusted to customize the detection and tracking sensitivity according to specific requirements.

## How to Use

1. **Dependencies Installation**: Install the required dependencies by running the following command:
    ```
    pip install -r requirements.txt
    ```

2. **Pre-trained Model**: Download the pre-trained YOLOv8 model [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

3. **Training Your Own Model**: Use the provided `TouristTraining.ipynb` Jupyter Notebook to train your own YOLOv8 model with custom data. This allows you to fine-tune the model according to your specific use case.

4. **Run the Script**: Execute the Python script using Python 3. Specify the path of the video you want to perform detection and tracking on.

5. **Adjust Parameters (Optional)**: Optionally, you can adjust the confidence and IoU thresholds to customize the detection and tracking sensitivity.

## Key Variables

- `model_name`: Name of the YOLOv8 model file used.
- `confidence_threshold`: Maximum confidence threshold for object detection.
- `iou_threshold`: Maximum IoU (Intersection over Union) threshold for non-maximum suppression.
- `video_name`: Name of the video file to perform detection and tracking on.
- `ZONE_POLYGON`: Area defined for detection on the video.
- `output_video_name`: Name of the output video file to be saved.

## Results

The adjusted and saved video file is named `output.mp4`, containing annotations indicating the detected tourists and their tracking IDs, along with safety zone alerts.

## Note

Ensure that all necessary dependencies are installed, and the pre-trained YOLOv8 model from Roboflow is placed in the correct location before running the code.

## License

This project is licensed under the terms of the [MIT License](LICENSE). For more details, refer to the LICENSE file.

