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
        
        logging.info(f"Plate detector initialized on {self.device} device")
    
    def detect_plate(self, image, object_id=None, event_data=None):
        """
        Detect license plate in an image
        Returns: cropped plate image or None
        """
        start_time = time.time()
        
        # Ensure image is in PIL format
        if not isinstance(image, Image.Image):
            try:
                # Convert from OpenCV
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logging.error(f"Failed to convert image format: {e}")
                return None
        
        # Use region of interest from prior detections
        roi = None
        if object_id and object_id in self.last_plates:
            # Get previous detection region
            prev_box = self.last_plates[object_id]['box']
            img_width, img_height = image.size
            
            # Expand ROI by 20%
            margin = 3.0
            x1, y1, x2, y2 = prev_box
            width, height = x2 - x1, y2 - y1
            
            # Apply margin while keeping within image bounds
            x1_roi = max(0, int(x1 - width * margin))
            y1_roi = max(0, int(y1 - height * margin))
            x2_roi = min(img_width, int(x2 + width * margin))
            y2_roi = min(img_height, int(y2 + height * margin))
            
            roi = (x1_roi, y1_roi, x2_roi, y2_roi)
            
            # Apply ROI if it's a reasonable size
            if (x2_roi - x1_roi) > 50 and (y2_roi - y1_roi) > 20:
                # Crop image to ROI for faster processing
                roi_image = image.crop((x1_roi, y1_roi, x2_roi, y2_roi))
                
                # Run detection on ROI
                try:
                    roi_result = self.model.predict(
                        roi_image,
                        device=self.device,
                        verbose=False,
                        conf=0.25,
                        half=True
                    )
                    
                    if roi_result[0].boxes.xyxy.shape[0] > 0:
                        # Found plate in ROI
                        x1r, y1r, x2r, y2r = map(int, roi_result[0].boxes.xyxy[0])
                        
                        # Convert ROI coordinates back to full image
                        x1 = x1r + x1_roi
                        y1 = y1r + y1_roi
                        x2 = x2r + x1_roi
                        y2 = y2r + y1_roi
                        
                        # Update cache
                        self.last_plates[object_id] = {
                            'box': (x1, y1, x2, y2),
                            'timestamp': time.time()
                        }
                        
                        # Maintain cache size
                        if len(self.last_plates) > self.max_cache_size:
                            oldest_key = min(self.last_plates.keys(), 
                                           key=lambda k: self.last_plates[k]['timestamp'])
                            del self.last_plates[oldest_key]
                        
                        cropped_plate = image.crop((x1, y1, x2, y2))
                        self.total_detections += 1
                        self.detection_times.append(time.time() - start_time)
                        return cropped_plate
                    
                    # If ROI fails, fall back to full image detection
                except Exception as e:
                    logging.warning(f"ROI detection failed, falling back to full image: {e}")
        
        # Run detection on full image
        try:
            result = self.model.predict(
                image, 
                device=self.device,
                verbose=False,
                conf=0.25, 
                half=True
            )
            
            # Check if any plates detected
            if result[0].boxes.xyxy.shape[0] > 0:
                # Get highest confidence detection
                confidences = result[0].boxes.conf
                best_idx = int(torch.argmax(confidences))
                
                x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[best_idx])
                cropped_plate = image.crop((x1, y1, x2, y2))
                
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
                    
            # Use the optimized ROI detection if we have prior info
            plate_img = self.detect_plate(frame, object_id, event_data)
            
            if plate_img is not None:
                processing_time = time.time() - start_time
                logging.info(f"Multi-frame detection completed in {processing_time*1000:.1f}ms")
                return plate_img
        
        # If no plates found in multiple frames, try traditional single-frame detection
        # on the middle frame (usually has better quality)
        middle_idx = len(process_frames) // 2
        
        if middle_idx < len(process_frames):
            return self.detect_plate(process_frames[middle_idx], object_id, event_data)
        else:
            return None