from ultralytics import YOLO
import torch
import numpy as np
import cv2
from PIL import Image
import time
import logging

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
            logging.info("MPS enabled for plate detection")
        elif torch.cuda.is_available():
            device = "cuda"
            logging.info("CUDA enabled for plate detection")
        else:
            device = "cpu"
            logging.warning("No GPU available, using CPU for plate detection")
        
        # Allow overriding with CPU for stability
        self.device = device
        
        # Performance tracking
        self.total_detections = 0
        self.detection_times = []
        
        # Adaptive parameters
        self.confidence_threshold = 0.1  # Start with moderate confidence
        self.last_plates = {}  # Cache of recent plate regions by object_id
        self.max_cache_size = 100
        self.backlog = 0  # Track processing backlog
        
        # Blur parameters
        self.blur_kernel_size = (7, 7)  # Size of the blur kernel
        self.blur_sigma = 1.5  # Sigma value for Gaussian blur
        
        logging.info(f"Plate detector initialized on {self.device} device")
    
    def apply_blur_postprocessing(self, image):
        """Apply Gaussian blur to the plate image after detection"""
        try:
            # Convert PIL Image to numpy array for OpenCV processing
            img_np = np.array(image)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_np, self.blur_kernel_size, self.blur_sigma)
            
            # Convert back to PIL Image
            blurred_pil = Image.fromarray(blurred)
            
            return blurred_pil
        except Exception as e:
            logging.error(f"Error applying blur postprocessing: {e}")
            return image  # Return original image if blur fails
    
    def detect_plate(self, image, object_id=None, event_data=None):
        """
        Detect license plate in an image
        Returns: cropped plate image or None
        """
        start_time = time.time()
        
        # Ensure image is in PIL format - minimal conversion needed for model
        if not isinstance(image, Image.Image):
            try:
                # Convert from OpenCV
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logging.error(f"Failed to convert image format: {e}")
                return None
        
        # Run detection on full image - no ROI preprocessing
        try:
            result = self.model.predict(
                image, 
                device=self.device,
                verbose=False,
                conf=0.25, 
                half=False  # Disabled half-precision
            )
            
            # Check if any plates detected
            if result[0].boxes.xyxy.shape[0] > 0:
                # Get highest confidence detection
                confidences = result[0].boxes.conf
                best_idx = int(torch.argmax(confidences))
                
                x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[best_idx])
                cropped_plate = image.crop((x1, y1, x2, y2))
                
                # # Apply blur postprocessing to the cropped plate
                # processed_plate = self.apply_blur_postprocessing(cropped_plate)
                
                # Cache this detection if object_id provided
                if object_id:
                    self.last_plates[object_id] = {
                        'box': (x1, y1, x2, y2),
                        'timestamp': time.time()
                    }
                    
                    # Maintain cache size
                    if len(self.last_plates) > self.max_cache_size:
                        oldest_key = min(self.last_plates.keys(), 
                                       key=lambda k: self.last_plates[k]['timestamp'])
                        del self.last_plates[oldest_key]
                
                self.total_detections += 1
                self.detection_times.append(time.time() - start_time)
                return cropped_plate
            else:
                processing_time = time.time() - start_time
                if processing_time > 0.25:  # Long processing time
                    logging.info(f"Long detection time: {processing_time*1000:.1f}ms with no result")
                return None
        except Exception as e:
            logging.error(f"Error in license plate detection: {e}")
            return None
        
    def get_stats(self):
        """Return performance statistics"""
        if not self.detection_times:
            return {"avg_time": 0, "total_detections": 0, "device": self.device}
            
        stats = {
            "avg_time": sum(self.detection_times[-100:]) / len(self.detection_times[-100:]),
            "total_detections": self.total_detections,
            "device": self.device,
            "confidence": self.confidence_threshold,
            "backlog": self.backlog
        }
        return stats

    def detect_plate_multiframe(self, frames, object_id=None, event_data=None):
        """
        Detect license plate across multiple sequential frames
        Returns the first successfully detected plate image or None
        """
        if not frames or len(frames) == 0:
            return None
            
        start_time = time.time()
        
        # Process up to 3 frames maximum for efficiency
        process_frames = frames[:min(3, len(frames))]
        
        for frame in process_frames:
            # Ensure image is in PIL format
            if not isinstance(frame, Image.Image):
                try:
                    # Convert from OpenCV
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    logging.error(f"Failed to convert image format: {e}")
                    continue
                    
            plate_img = self.detect_plate(frame, object_id, event_data)
            
            if plate_img is not None:
                processing_time = time.time() - start_time
                logging.info(f"Multi-frame detection completed in {processing_time*1000:.1f}ms")
                return plate_img, 
        
        # If no plates found in multiple frames, try one more frame
        middle_idx = len(process_frames) // 2
        
        if middle_idx < len(process_frames):
            return self.detect_plate(process_frames[middle_idx], object_id, event_data)
        else:
            return None