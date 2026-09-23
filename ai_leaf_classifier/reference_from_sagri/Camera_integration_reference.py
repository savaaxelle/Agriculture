#!/usr/bin/env python3
"""
Real-time Plant Disease Detection with Camera Integration
Uses the trained EfficientNetB2 model for live plant disease detection
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import time
import os

# Model configuration
num_classes = 5
class_names = ['Guava - Disease Free', 'Guava - Phytopthora', 'Guava - Red rust', 'Guava - Scab', 'Guava - Styler and Root']

# Confidence threshold for guava detection
GUAVA_CONFIDENCE_THRESHOLD = 0.6  # Lowered from 0.8 to 0.6 (60% minimum confidence)

class PlantDiseaseCamera:
    def __init__(self, model_path):
        """Initialize the camera-based plant disease detector."""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize camera variables
        self.cap = None
        self.running = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Camera detector initialized successfully!")
    
    def load_model(self, model_path):
        """Load the trained EfficientNetB2 model."""
        try:
            # Load the model architecture with updated syntax
            model = models.efficientnet_b2(weights=None)  # Use weights=None instead of pretrained=False
            # Modify the final classifier layer to match our number of classes
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(1408, num_classes)  # EfficientNet-B2's last layer has 1408 input features
            )
            
            # Load the trained weights directly (not wrapped in checkpoint)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded from {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for model inference with low-resolution enhancement."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            height, width = image_rgb.shape[:2]
            
            # Enhance low-resolution images
            if width < 224 or height < 224:
                # Upscale using INTER_CUBIC for better quality
                scale_factor = max(224 / width, 224 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Apply slight sharpening to enhance details
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                image_rgb = cv2.filter2D(image_rgb, -1, kernel)
                
                # Apply bilateral filter to reduce noise while preserving edges
                image_rgb = cv2.bilateralFilter(image_rgb, 9, 75, 75)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transformations
            tensor = self.transform(pil_image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Make prediction on the given image with adaptive thresholds for low-resolution images."""
        try:
            # Get image dimensions for adaptive thresholds
            height, width = image.shape[:2]
            image_area = height * width
            
            # Adjust confidence threshold based on image quality
            if image_area < 50000:  # Very small images (< 50k pixels)
                adaptive_threshold = 0.35  # Lowered from 0.4 to 0.35 for very small images
            elif image_area < 100000:  # Small images (< 100k pixels)
                adaptive_threshold = 0.5  # Medium threshold
            else:
                adaptive_threshold = GUAVA_CONFIDENCE_THRESHOLD  # Use default threshold
            
            # Preprocess the image
            tensor = self.preprocess_image(image)
            if tensor is None:
                return "Error", 0.0, [0.0] * num_classes, False
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                all_probabilities = probabilities[0].cpu().numpy()
            
            # Check if this is likely a guava plant
            max_prob = max(all_probabilities)
            second_max_prob = sorted(all_probabilities)[-2]
            probability_gap = max_prob - second_max_prob
            
            # Adaptive conditions for guava detection based on image quality
            if image_area < 50000:
                # Very lenient for very small images
                is_guava = (confidence >= adaptive_threshold and 
                           probability_gap >= 0.05 and  # Very small gap
                           confidence > 0.2)  # Very low safety check
            elif image_area < 100000:
                # Medium leniency for small images
                is_guava = (confidence >= adaptive_threshold and 
                           probability_gap >= 0.08 and  # Small gap
                           confidence > 0.25)  # Low safety check
            else:
                # Standard conditions for normal images
                is_guava = (confidence >= adaptive_threshold and 
                           probability_gap >= 0.1 and  # Normal gap
                           confidence > 0.3)  # Normal safety check
            
            return class_names[predicted_class_idx], confidence, all_probabilities, is_guava
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return "Error", 0.0, [0.0] * num_classes, False
    
    def start_camera(self, camera_index=0, width=640, height=480):
        """Start the camera capture."""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera at index {camera_index}")
            
            print(f"✅ Camera started: {width}x{height}")
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            raise
    
    def stop_camera(self):
        """Stop the camera capture."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera stopped")
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_frame(self, frame):
        """Save the current frame."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"guava_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved as {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")