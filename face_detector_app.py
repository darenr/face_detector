#!/usr/bin/env python3
"""
Face Detector Gradio Application

This module provides a Gradio interface for the FaceDetectorApp class,
allowing users to detect faces in both uploaded images and webcam streams.
"""

import glob
import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import sys
import gradio as gr
from loguru import logger

from face_detector import FaceDetectorApp


# add logger to stderr
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    colorize=True,
    backtrace=True,
    diagnose=True,
)


class FaceDetectorGradioApp:
    """
    Gradio interface for the FaceDetectorApp.

    This class wraps the FaceDetectorApp with a Gradio interface for
    interactive face detection on both images and webcam streams.
    """

    def __init__(
        self,
        default_min_face_size: int = 20,
        default_confidence: float = 0.7,
        device: Optional[str] = None,
        theme: str = "soft",
    ) -> None:
        """
        Initialize the Gradio application.

        Args:
            default_min_face_size: Default minimum face size to detect
            default_confidence: Default confidence threshold
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            theme: Gradio theme
        """
        logger.info("Initializing FaceDetectorGradioApp")

        self.default_min_face_size = default_min_face_size
        self.default_confidence = default_confidence
        self.theme = theme

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize face detector with default settings
        self.detector = FaceDetectorApp(
            min_confidence=default_confidence,
            min_face_size=default_min_face_size,
            device=self.device,
        )

        # Initialize Gradio interface
        self.interface = self._build_interface()

        logger.info("FaceDetectorGradioApp initialized successfully")

    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio interface with tabs for image and webcam."""
        with gr.Blocks(theme=self.theme, title="Face Detector") as interface:
            gr.Markdown("# Face Detector Application")

            with gr.Tabs() as tabs:  # noqa: F841
                # Image tab
                with gr.Tab("Image Upload"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Input controls for image
                            image_input = gr.Image(type="numpy", label="Upload Image")

                            with gr.Row():
                                image_min_face = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=self.default_min_face_size,
                                    step=5,
                                    label="Minimum Face Size (pixels)",
                                )
                                image_confidence = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=self.default_confidence,
                                    step=0.05,
                                    label="Confidence Threshold",
                                )

                            image_detect_btn = gr.Button("Detect Faces", variant="primary")

                        with gr.Column(scale=1):
                            # Output for image
                            image_output = gr.Image(type="numpy", label="Detection Result")
                            image_json = gr.JSON(label="Detection Details")
                            image_stats = gr.Textbox(label="Performance Stats")

                # Webcam tab
                with gr.Tab("Webcam"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Input controls for webcam
                            webcam_input = gr.Image(
                                type="numpy",
                                sources="webcam",
                                streaming=True,  # Changed from True to False for more reliable frame capture
                                label="Webcam Feed",
                                mirror_webcam=True,  # Mirror the webcam for a more natural experience
                            )

                            with gr.Row():
                                webcam_min_face = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=self.default_min_face_size,
                                    step=5,
                                    label="Minimum Face Size (pixels)",
                                )
                                webcam_confidence = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=self.default_confidence,
                                    step=0.05,
                                    label="Confidence Threshold",
                                )

                            # Add a button to process webcam frames
                            webcam_detect_btn = gr.Button("Process Frame", variant="primary")

                            # Add instructions for webcam usage
                            gr.Markdown("""
                            ### Instructions:
                            1. Allow camera access when prompted
                            2. Take a snapshot with the webcam button
                            3. Click 'Process Frame' to detect faces
                            """)

                        with gr.Column(scale=1):
                            # Output for webcam
                            webcam_output = gr.Image(type="numpy", label="Detection Result")
                            webcam_json = gr.JSON(label="Detection Details")
                            webcam_stats = gr.Textbox(label="Performance Stats")

            # Event handlers
            image_detect_btn.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    image_min_face,
                    image_confidence,
                ],
                outputs=[image_output, image_json, image_stats],
            )

            # Replace webcam_input.change with webcam_detect_btn.click
            webcam_detect_btn.click(
                fn=self.process_webcam_frame,
                inputs=[
                    webcam_input,
                    webcam_min_face,
                    webcam_confidence,
                ],
                outputs=[webcam_output, webcam_json, webcam_stats],
            )

            example_images = glob.glob("examples/*.jpg")
            logger.info(f"Found {len(example_images)} example images")

            # Examples for image tab
            gr.Examples(
                examples=example_images,
                inputs=image_input,
                outputs=[image_output, image_json, image_stats],
                fn=lambda x: self.process_image(
                    x,
                    self.default_min_face_size,
                    self.default_confidence,
                ),
                cache_examples=False,
            )

        return interface

    def update_detector_settings(
        self, min_face_size: int, confidence: float, select_largest: bool = False
    ) -> None:
        """
        Update detector settings based on UI inputs.

        Args:
            min_face_size: Minimum face size to detect
            confidence: Confidence threshold
            select_largest: Whether to select only the largest face (default: False)
        """
        logger.info(
            f"Updating detector settings: min_face_size={min_face_size}, confidence={confidence}, select_largest={select_largest}"
        )

        # Reinitialize detector with new settings
        self.detector = FaceDetectorApp(
            min_face_size=min_face_size,
            min_confidence=confidence,
            device=self.device,
            keep_all=not select_largest,  # If select_largest is True, keep_all should be False
        )

    def process_image(
        self,
        image: Optional[np.ndarray],
        min_face_size: int,
        confidence: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """
        Process an uploaded image for face detection.

        Args:
            image: Input image as numpy array
            min_face_size: Minimum face size to detect
            confidence: Confidence threshold

        Returns:
            Tuple containing:
                - Annotated image with face detections
                - JSON data with detection results
                - Performance statistics text
        """
        if image is None:
            return None, [], "No image provided"

        # Log image information for debugging
        logger.info(f"Received image: shape={image.shape}, dtype={image.dtype}")

        # Update detector settings if changed
        if (
            min_face_size != self.detector.min_face_size
            or confidence != self.detector.min_confidence
        ):
            self.update_detector_settings(min_face_size, confidence, False)

        try:
            # Convert RGB to BGR (OpenCV expects BGR format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Detect faces
            start_time = time.time()
            faces = self.detector.detect_faces(image_bgr)
            detection_time = (time.time() - start_time) * 1000  # ms

            # Draw faces on image
            result_bgr = self.detector.draw_faces(
                image_bgr,
                faces,
            )

            # Convert back to RGB for display
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            # Prepare performance stats
            stats = f"Detected {len(faces)} faces in {detection_time:.2f}ms"

            logger.info(f"Image processing complete: {stats}")
            return result_rgb, faces, stats

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return image, [], f"Error: {str(e)}"

    def process_webcam_frame(
        self,
        frame: Optional[np.ndarray],
        min_face_size: int,
        confidence: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """
        Process a webcam frame for face detection.

        Args:
            frame: Input frame as numpy array
            min_face_size: Minimum face size to detect
            confidence: Confidence threshold

        Returns:
            Tuple containing:
                - Annotated frame with face detections
                - JSON data with detection results
                - Performance statistics text
        """
        if frame is None:
            logger.error("Webcam frame is None. Check webcam connection.")
            return None, [], "No frame provided"

        # Log frame information for debugging
        logger.info(f"Received webcam frame: shape={frame.shape}, dtype={frame.dtype}")

        # Update detector settings if changed
        if (
            min_face_size != self.detector.min_face_size
            or confidence != self.detector.min_confidence
        ):
            # Use lower confidence threshold for webcam
            webcam_confidence = max(0.3, confidence - 0.2)
            self.update_detector_settings(min_face_size, webcam_confidence, False)
            logger.info(f"Lowered confidence threshold for webcam to {webcam_confidence}")

        try:
            # Ensuring correct color format for face detection
            # Gradio webcam gives RGB format, we need to convert to BGR for OpenCV processing
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                logger.info("Converted webcam frame from RGB to BGR")
            else:
                frame_bgr = frame
                logger.info("Using webcam frame as is (not RGB)")

            # Apply preprocessing to enhance face detection for webcam
            # 1. Resize if the image is too large (better performance, sometimes better detection)
            height, width = frame_bgr.shape[:2]
            max_dimension = 640  # Maximum dimension for processing

            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_width, new_height))
                logger.info(f"Resized webcam frame to {new_width}x{new_height}")

            # 2. Apply histogram equalization to improve contrast
            if len(frame_bgr.shape) == 3:
                # Convert to YUV and equalize the Y channel
                frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
                frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
                frame_bgr_eq = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
                logger.info("Applied histogram equalization")
            else:
                frame_bgr_eq = frame_bgr

            # Keep both original and enhanced versions
            # Try detection on enhanced version first
            start_time = time.time()
            faces = self.detector.detect_faces(frame_bgr_eq)

            # If no faces detected in enhanced version, try original
            if len(faces) == 0:
                logger.info("No faces detected in enhanced frame, trying original frame")
                faces = self.detector.detect_faces(frame_bgr)

            detection_time = (time.time() - start_time) * 1000  # ms

            logger.info(f"Face detection results: Found {len(faces)} faces")

            # Draw faces on original frame (not the equalized one for better appearance)
            result_bgr = self.detector.draw_faces(
                frame_bgr,
                faces,
            )

            # Convert back to RGB for display in Gradio
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            # Prepare performance stats
            stats = f"Detected {len(faces)} faces in {detection_time:.2f}ms"

            logger.info(f"Webcam processing complete: {stats}")
            return result_rgb, faces, stats

        except Exception as e:
            logger.error(f"Error processing webcam frame: {str(e)}")
            logger.exception("Stack trace:")
            return frame, [], f"Error: {str(e)}"

    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Launch the Gradio interface.

        Args:
            server_name: Server hostname
            server_port: Server port
            share: Whether to create a public link
            debug: Whether to run in debug mode
        """
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        self.interface.launch(
            server_name=server_name, server_port=server_port, share=share, debug=debug
        )


# Create and launch app when run directly
if __name__ == "__main__":
    # Launch the app
    app = FaceDetectorGradioApp()
    app.launch()
