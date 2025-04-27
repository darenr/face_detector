#!/usr/bin/env python3
"""
Face Detector Module using PyTorch

This module provides a standalone face detection class using PyTorch-based models.
It can detect faces in both images and video streams, with optimized performance
and reduced false positives.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
import sys
from facenet_pytorch import MTCNN
from loguru import logger


class FaceDetectorApp:
    """
    A class for detecting faces in images and video streams using PyTorch.

    This class provides methods to detect faces in both static images and
    webcam video streams. It returns bounding boxes as percentages of the
    original image dimensions and is optimized for performance and reduced
    false positives.
    """

    def __init__(
        self,
        min_face_size: int = 20,
        min_confidence: float = 0.7,
        device: Optional[str] = None,
        keep_all: bool = True,
        post_process: bool = True,
        margin: int = 0,
    ) -> None:
        """
        Initialize the FaceDetectorApp.

        Args:
            min_face_size: Minimum face size to detect (in pixels)
            min_confidence: Minimum confidence threshold for face detection (0.0 to 1.0)
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            keep_all: Whether to keep all detected faces (True) or only the most confident one (False)
            post_process: Whether to apply post-processing to bounding boxes
            margin: Margin to add around detected faces (in pixels)
        """
        logger.info(f"Initializing FaceDetectorApp with min_confidence: {min_confidence}")

        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.keep_all = keep_all
        self.post_process = post_process
        self.margin = margin

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize face detector
        self.detector = None
        self._initialize_detector()

        # Performance metrics
        self.last_detection_time = 0
        self.frame_count = 0
        self.total_detection_time = 0

        logger.info("FaceDetectorApp initialized successfully")

    def _initialize_detector(self) -> None:
        """Initialize the PyTorch-based face detector."""
        try:
            logger.info(
                f"Loading MTCNN face detector model with min_face_size: {self.min_face_size}"
            )
            self.detector = MTCNN(
                image_size=160,
                margin=self.margin,
                min_face_size=self.min_face_size,
                thresholds=[0.5, 0.6, 0.7],  # Lower thresholds to detect more faces
                factor=0.709,  # Scale factor for image pyramid
                post_process=self.post_process,
                device=self.device,
                keep_all=self.keep_all,
                select_largest=False,
            )
            logger.info("Face detector model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            raise

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image and return bounding boxes as percentages.

        Args:
            img: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of dictionaries containing face detection results:
                - 'confidence': Detection confidence score
                - 'region': Dictionary with percentage-based coordinates:
                    - 'x': Left coordinate as percentage of image width
                    - 'y': Top coordinate as percentage of image height
                    - 'w': Width as percentage of image width
                    - 'h': Height as percentage of image height
                - 'time_ms': Detection time in milliseconds
        """
        if img is None or img.size == 0:
            logger.error("Empty or invalid image provided")
            return []

        # Record start time for performance measurement
        start_time = time.time()

        # Get image dimensions for percentage calculation
        img_h, img_w = img.shape[:2]

        try:
            # Convert BGR to RGB (PyTorch models expect RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            # Returns: boxes, probs, landmarks
            boxes, probs, landmarks = self.detector.detect(img_rgb, landmarks=True)

            faces = []

            # Process detection results if faces were found
            if boxes is not None and len(boxes) > 0:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    # Filter by confidence to reduce false positives
                    if prob >= self.min_confidence:
                        # Get the facial area coordinates
                        x1, y1, x2, y2 = box

                        # Calculate width and height
                        w = x2 - x1
                        h = y2 - y1

                        # Convert to percentages of image dimensions
                        x_pct = (x1 / img_w) * 100
                        y_pct = (y1 / img_h) * 100
                        w_pct = (w / img_w) * 100
                        h_pct = (h / img_h) * 100

                        # Get facial landmarks if available
                        face_landmarks = None
                        if landmarks is not None and i < len(landmarks):
                            face_landmarks = landmarks[i].tolist()

                        faces.append(
                            {
                                "confidence": float(prob),
                                "region": {
                                    "x": float(x_pct),
                                    "y": float(y_pct),
                                    "w": float(w_pct),
                                    "h": float(h_pct),
                                },
                                "original_region": {
                                    "x": int(x1),
                                    "y": int(y1),
                                    "w": int(w),
                                    "h": int(h),
                                },
                                "landmarks": face_landmarks,
                            }
                        )

            # Calculate detection time
            detection_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update performance metrics
            self.frame_count += 1
            self.total_detection_time += detection_time
            self.last_detection_time = detection_time

            # Add detection time to each face
            for face in faces:
                face["time_ms"] = detection_time

            logger.info(f"Detected {len(faces)} faces in {detection_time:.2f}ms")
            return faces

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []

    def draw_faces(
        self,
        img: np.ndarray,
        faces: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw bounding boxes, confidence scores, and landmarks on the image.

        Args:
            img: Input image as numpy array
            faces: List of face detection results from detect_faces()
            color: BGR color tuple for bounding box
            thickness: Line thickness for bounding box
            show_confidence: Whether to display confidence scores
            font_scale: Font scale for confidence text

        Returns:
            Image with drawn bounding boxes and optional landmarks
        """
        img_h, img_w = img.shape[:2]
        result_img = img.copy()

        for face in faces:
            # Get percentage-based region
            region = face["region"]

            # Convert percentages back to pixel coordinates
            x = int((region["x"] / 100) * img_w)
            y = int((region["y"] / 100) * img_h)
            w = int((region["w"] / 100) * img_w)
            h = int((region["h"] / 100) * img_h)

            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)

            # Show confidence if requested
            if show_confidence and "confidence" in face:
                confidence = face["confidence"]
                text = f"{confidence:.2f}"
                cv2.putText(
                    result_img,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                )

        return result_img

    def process_webcam(
        self,
        camera_id: int = 0,
        display_window: bool = True,
        window_name: str = "Face Detection",
        fps_limit: Optional[int] = 15,
    ) -> None:
        """
        Process webcam video stream for face detection.

        Args:
            camera_id: Camera device ID
            display_window: Whether to display the video window
            window_name: Name of the display window
            fps_limit: Maximum frames per second (None for no limit)
        """
        logger.info(f"Starting webcam processing with camera ID: {camera_id}")

        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera with ID: {camera_id}")
            return

        # Calculate minimum time between frames if FPS limit is set
        min_frame_time = 0
        if fps_limit is not None and fps_limit > 0:
            min_frame_time = 1.0 / fps_limit

        last_frame_time = time.time()

        try:
            while True:
                # Respect FPS limit if set
                if fps_limit is not None:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < min_frame_time:
                        # Sleep to maintain FPS limit
                        time.sleep(min_frame_time - elapsed)

                # Capture frame
                last_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break

                # Detect faces
                faces = self.detect_faces(frame)

                # Draw faces on frame
                result_frame = self.draw_faces(frame, faces)

                # Add performance metrics to frame
                avg_time = self.total_detection_time / max(1, self.frame_count)
                cv2.putText(
                    result_frame,
                    f"Avg: {avg_time:.1f}ms, Last: {self.last_detection_time:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                # Display the frame
                if display_window:
                    cv2.imshow(window_name, result_frame)

                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Webcam processing interrupted by user")
        except Exception as e:
            logger.error(f"Error in webcam processing: {str(e)}")
        finally:
            # Release resources
            cap.release()
            if display_window:
                cv2.destroyAllWindows()

            logger.info("Webcam processing stopped")

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for face detection.

        Returns:
            Dictionary with performance metrics:
                - 'avg_detection_time_ms': Average detection time in milliseconds
                - 'last_detection_time_ms': Last detection time in milliseconds
                - 'frame_count': Number of frames processed
        """
        avg_time = self.total_detection_time / max(1, self.frame_count)

        return {
            "avg_detection_time_ms": avg_time,
            "last_detection_time_ms": self.last_detection_time,
            "frame_count": self.frame_count,
        }


# Example usage
if __name__ == "__main__":
    # Create face detector
    detector = FaceDetectorApp(
        min_confidence=0.7,
        device="cpu",  # Use 'cuda' for GPU acceleration if available
    )

    # Process webcam feed
    detector.process_webcam(camera_id=0)
