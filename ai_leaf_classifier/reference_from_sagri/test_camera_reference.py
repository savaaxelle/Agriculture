#!/usr/bin/env python3
"""
Real-time Guava Disease Detection Camera Test
Similar to test_image.py but for live camera feed
Just change the camera settings at the bottom to test different configurations
"""

import os
import sys
import numpy as np
import cv2

# Add the current directory to the path so we can import Camera_integration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Camera_integration import PlantDiseaseCamera

def test_camera(camera_index=0, width=640, height=480):
    """Test the guava disease detection on live camera feed."""
    
    print(f"🔍 Testing camera: Index {camera_index}, Resolution {width}x{height}")
    print("=" * 50)
    
    detector = None  # Initialize detector variable
    
    try:
        # Initialize the detector
        detector = PlantDiseaseCamera(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'efficientnet_b2.pt'))
        
        # Start camera
        detector.start_camera(camera_index, width, height)
        detector.running = True
        
        print("📹 Camera started successfully!")
        print("Controls: Q-Quit, S-Save frame, R-Reset FPS counter")
        print("-" * 50)
        
        while detector.running:
            if detector.cap is None:
                print("❌ Camera not initialized")
                break
                
            ret, frame = detector.cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            prediction, confidence, all_probs, is_guava = detector.predict(frame)
            
            # Additional strict filtering for camera detection
            # Check if the highest probability is significantly higher than others
            max_prob = max(all_probs)
            second_max_prob = sorted(all_probs)[-2]
            probability_gap = max_prob - second_max_prob
            
            # More strict conditions for camera detection
            is_guava = (is_guava and 
                       confidence >= 0.8 and  # At least 80% confidence
                       probability_gap >= 0.3)  # At least 30% gap between top predictions
            
            # Update FPS
            detector.update_fps()
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Create a clean background for text overlay
            height, width = frame.shape[:2]
            text_width = 400
            text_height = height
            text_image = np.ones((text_height, text_width, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw text information on the right side
            y_offset = 60
            line_height = 30
            
            # Title with beautiful styling
            cv2.putText(text_image, "GUAVA DISEASE DETECTION", (30, int(y_offset)), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 50, 150), 2)  # Bold title
            y_offset += line_height * 2
            
            # Detection result with beautiful colors
            if is_guava:
                # Success message in green
                cv2.putText(text_image, "GUAVA PLANT DETECTED", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 120, 0), 2)  # Bold text
                y_offset += line_height * 1.5
                
                # Disease prediction with highlight
                cv2.putText(text_image, "DISEASE CLASSIFICATION:", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (50, 50, 50), 2)  # Bold label
                y_offset += line_height
                
                cv2.putText(text_image, f"{prediction}", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 80, 150), 2)  # Bold disease name
                y_offset += line_height * 1.5
                
                # Confidence with progress bar
                cv2.putText(text_image, f"CONFIDENCE: {confidence:.1%}", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (50, 50, 50), 2)  # Bold confidence
                y_offset += line_height
                
                # Draw confidence bar
                bar_width = int(confidence * 200)
                cv2.rectangle(text_image, (30, int(y_offset)), (230, int(y_offset) + 15), (200, 200, 200), 2)
                cv2.rectangle(text_image, (30, int(y_offset)), (30 + bar_width, int(y_offset) + 15), (0, 120, 0), -1)
                y_offset += line_height * 2
            else:
                # No detection message in red
                cv2.putText(text_image, "NO GUAVA PLANT DETECTED", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 0, 0), 2)  # Bold text
                y_offset += line_height * 1.5
                
                cv2.putText(text_image, f"MAXIMUM CONFIDENCE: {confidence:.1%}", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (50, 50, 50), 2)  # Bold confidence
                y_offset += line_height * 2
            
            # All probabilities with beautiful styling (only show when not paused)
            cv2.putText(text_image, "DISEASE PROBABILITIES:", (30, int(y_offset)), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (50, 50, 50), 2)  # Slightly bigger than probabilities
            y_offset += line_height * 1.5
            
            from Camera_integration import class_names
            for class_name, prob in zip(class_names, all_probs):
                disease_name = class_name.replace("Guava - ", "")
                # Highlight the predicted class with color only, not bold
                if class_name == prediction:
                    color = (0, 120, 0)  # Green for predicted
                    thickness = 1  # Normal thickness for all
                else:
                    color = (80, 80, 80)  # Gray for others
                    thickness = 1  # Normal thickness for all
                
                cv2.putText(text_image, f"{disease_name}: {prob:.1%}", (30, int(y_offset)), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, color, thickness)
                y_offset += line_height
            
            # Add FPS display
            cv2.putText(text_image, f"FPS: {detector.current_fps:.1f}", (30, text_height - 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (50, 50, 50), 1)
            
            # Resize frame to match text panel height
            resized_frame = cv2.resize(frame, (text_height, text_height))
            
            # Combine frame and text side by side
            combined_image = np.hstack([resized_frame, text_image])
            
            # Display the combined image
            cv2.imshow('Guava Disease Detection - Live Camera', combined_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detector.save_frame(frame)
            elif key == ord('r'):
                detector.fps_counter = 0
                detector.fps_start_time = detector.fps_start_time
                print("🔄 FPS counter reset")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping camera detection...")
    except Exception as e:
        print(f"❌ Error during camera testing: {e}")
    finally:
        if detector is not None:
            detector.stop_camera()
        print("📹 Camera detection stopped.")

if __name__ == "__main__":
    # ===========================================
    # CHANGE CAMERA SETTINGS HERE TO TEST DIFFERENT CONFIGURATIONS
    # ===========================================
    camera_index = 0  # Change this to test different cameras (0, 1, 2, etc.)
    width = 640       # Change this for different resolutions
    height = 480      # Change this for different resolutions
    
    # Examples:
    # camera_index = 0, width = 640, height = 480   # Standard webcam
    # camera_index = 1, width = 1280, height = 720  # External camera, HD
    # camera_index = 0, width = 1920, height = 1080 # Full HD
    # ===========================================
    
    test_camera(camera_index, width, height) 