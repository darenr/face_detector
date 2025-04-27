#!/usr/bin/env python3
"""
Test script for the face detector application.

This script tests the face detector class with a sample image.
"""

import os
import cv2
import numpy as np
from face_detector import FaceDetectorApp


def test_image_detection():
    """Test face detection on a sample image."""
    print("Testing face detection on sample image...")

    # Create detector
    detector = FaceDetectorApp(min_confidence=0.7, device="cpu")

    # Load test image
    image_path = "examples/example1.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return False

    # Detect faces
    faces = detector.detect_faces(img)

    # Draw faces on image
    result_img = detector.draw_faces(img, faces)

    # Save result
    output_path = "examples/result1.jpg"
    cv2.imwrite(output_path, result_img)

    print(f"Detected {len(faces)} faces in the image")
    print(f"Result saved to {output_path}")

    # Print face details
    for i, face in enumerate(faces):
        print(f"Face {i + 1}:")
        print(f"  Confidence: {face['confidence']:.2f}")
        print(
            f"  Region (% of image): x={face['region']['x']:.1f}%, y={face['region']['y']:.1f}%, w={face['region']['w']:.1f}%, h={face['region']['h']:.1f}%"
        )
        print(f"  Detection time: {face.get('time_ms', 0):.2f}ms")

    # Test performance stats
    stats = detector.get_performance_stats()
    print("\nPerformance stats:")
    print(f"  Average detection time: {stats['avg_detection_time_ms']:.2f}ms")
    print(f"  Last detection time: {stats['last_detection_time_ms']:.2f}ms")
    print(f"  Frame count: {stats['frame_count']}")

    return len(faces) > 0


def test_multiple_images():
    """Test face detection on multiple images."""
    print("\nTesting face detection on multiple images...")

    # Create detector
    detector = FaceDetectorApp(min_confidence=0.7, device="cpu")

    # Test images
    image_paths = ["examples/example1.jpg", "examples/example2.jpg"]

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Error: Test image not found at {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            continue

        # Detect faces
        faces = detector.detect_faces(img)

        # Draw faces on image
        result_img = detector.draw_faces(img, faces)

        # Save result
        output_path = f"examples/result{i + 1}.jpg"
        cv2.imwrite(output_path, result_img)

        print(f"Image {i + 1}: Detected {len(faces)} faces")
        print(f"Result saved to {output_path}")


if __name__ == "__main__":
    print("Testing PyTorch Face Detector")
    print("-" * 40)

    success = test_image_detection()
    if success:
        test_multiple_images()
        print("\nAll tests completed successfully!")
    else:
        print("\nTest failed: No faces detected in sample image")
