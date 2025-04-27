# Face Detection Application - Usage Instructions

This document provides instructions for using the PyTorch-based face detection application.

## Overview

This application provides face detection capabilities for both images and webcam streams using PyTorch and the MTCNN model from the facenet-pytorch library. The application includes:

1. A standalone `FaceDetectorApp` class that can be imported and used in other Python projects
2. A Gradio web interface for interactive testing and demonstration

The face detector returns bounding boxes as percentages of the original image dimensions, is optimized for performance, and includes features to reduce false positives.

## Requirements

- Python 3.6+
- PyTorch
- facenet-pytorch
- OpenCV
- Gradio
- Loguru (for logging)

All dependencies can be installed using:

```bash
pip install torch torchvision facenet-pytorch opencv-python numpy typing-extensions loguru gradio
```

## File Structure

- `face_detector.py`: The standalone face detector class
- `face_detector_app.py`: The Gradio web interface
- `test_face_detector.py`: Test script for the face detector class
- `examples/`: Directory containing example images and test results

## Using the Standalone Face Detector Class

The `FaceDetectorApp` class in `face_detector.py` can be imported and used in your own Python projects:

```python
from face_detector import FaceDetectorApp

# Create a face detector instance
detector = FaceDetectorApp(
    min_face_size=20,        # Minimum face size to detect (pixels)
    min_confidence=0.7,      # Confidence threshold (0.0 to 1.0)
    device='cpu',            # 'cpu' or 'cuda' for GPU acceleration
    keep_all=True,           # Keep all detected faces
    select_largest=False     # Whether to select only the largest face
)

# Detect faces in an image
import cv2
image = cv2.imread('path/to/image.jpg')
faces = detector.detect_faces(image)

# Draw bounding boxes on the image
result_image = detector.draw_faces(
    image, 
    faces,
    show_landmarks=True,     # Whether to show facial landmarks
    show_confidence=True     # Whether to show confidence scores
)

# Save or display the result
cv2.imwrite('result.jpg', result_image)
cv2.imshow('Face Detection', result_image)
cv2.waitKey(0)

# Process webcam feed
detector.process_webcam(
    camera_id=0,             # Camera device ID
    display_window=True,     # Whether to display the video window
    fps_limit=15             # Maximum frames per second
)

# Get performance statistics
stats = detector.get_performance_stats()
print(f"Average detection time: {stats['avg_detection_time_ms']}ms")
```

### Face Detection Results

The `detect_faces()` method returns a list of dictionaries, each containing:

- `confidence`: Detection confidence score (0.0 to 1.0)
- `region`: Dictionary with percentage-based coordinates:
  - `x`: Left coordinate as percentage of image width
  - `y`: Top coordinate as percentage of image height
  - `w`: Width as percentage of image width
  - `h`: Height as percentage of image height
- `original_region`: Dictionary with pixel-based coordinates
- `landmarks`: Facial landmarks (if available)
- `time_ms`: Detection time in milliseconds

## Using the Gradio Web Interface

The Gradio interface provides an easy way to test the face detector with both image uploads and webcam streams.

### Running the Interface

```bash
python face_detector_app.py
```

This will start the Gradio web server, typically on http://localhost:7860.

### Interface Features

The interface has two tabs:

1. **Image Upload**: Upload and process images
   - Upload an image
   - Adjust minimum face size and confidence threshold
   - Toggle facial landmarks display
   - Toggle largest face selection
   - Click "Detect Faces" to process the image

2. **Webcam**: Process webcam stream
   - Allow webcam access in your browser
   - Adjust minimum face size and confidence threshold
   - Toggle facial landmarks display
   - Toggle largest face selection
   - Webcam feed is processed in real-time

### Output

Both tabs provide:

- Processed image/frame with bounding boxes
- JSON data with detection details
- Performance statistics

## Performance Optimization

For better performance:

1. Use a GPU if available by setting `device='cuda'`
2. Increase the minimum face size for faster processing
3. Adjust the confidence threshold to balance between detection accuracy and false positives
4. Use the `select_largest=True` option when only the main face is needed

## Logging

The application uses the Loguru library for logging. Logs are written to:

- Console output
- `face_detector.log` for the detector class
- `gradio_app.log` for the Gradio interface

## Troubleshooting

- If no faces are detected, try:
  - Decreasing the confidence threshold
  - Decreasing the minimum face size
  - Ensuring the image has sufficient lighting and resolution
- If the application is slow:
  - Increase the minimum face size
  - Use a GPU if available
  - Limit the FPS for webcam processing
- If you encounter CUDA errors:
  - Force CPU mode with `device='cpu'`
