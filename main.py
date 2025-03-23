import time
import threading
import os
import json
import numpy as np
from PIL import Image
import cv2
import logging
import queue
from typing import Dict, Any

from detection_log_processor import DetectionLogProcessor
from plate_detector import PlateDetector
from ocr_plate import OCRPlate
from frame_buffer import FrameBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- Load configuration from config.json ---
with open("config.json", "r") as f:
    config = json.load(f)

os.makedirs(config["output_dir"], exist_ok=True)

# Global logs and locks
detection_log_list = []
processed_results = []
file_lock = threading.Lock()

# Performance metrics
acquisition_times = []
processing_times = []
METRICS_MAX_SAMPLES = 100
metrics_lock = threading.Lock()

# Shutdown event for clean termination
shutdown_event = threading.Event()

# Create the shared frame buffer between pipelines
frame_buffer = FrameBuffer(max_size=config.get("buffer_size", 30))

# Global reference to the frame reader
rtsp_reader = None
detection_processor = None

def update_metrics(times_list, new_time):
    """Update processing time metrics"""
    with metrics_lock:
        times_list.append(new_time)
        if len(times_list) > METRICS_MAX_SAMPLES:
            times_list.pop(0)

def log_metrics():
    """Log performance metrics for both pipelines"""
    with metrics_lock:
        if acquisition_times:
            avg_acquisition = sum(acquisition_times) / len(acquisition_times)
            max_acquisition = max(acquisition_times)
            logging.info(f"Acquisition Pipeline: Avg={avg_acquisition:.3f}s, Max={max_acquisition:.3f}s")
        
        if processing_times:
            avg_processing = sum(processing_times) / len(processing_times)
            max_processing = max(processing_times)
            logging.info(f"Processing Pipeline: Avg={avg_processing:.3f}s, Max={max_processing:.3f}s")

# Pipeline 1: Frame Acquisition
class FrameAcquisitionPipeline:
    """
    Pipeline 1: Continuously acquires frames from RTSP and handles MQTT events
    """
    def __init__(self, rtsp_url):
        global rtsp_reader
        
        # Initialize RTSP reader
        from rtsp_frame_reader import RTSPFrameReader
        rtsp_reader = RTSPFrameReader(rtsp_url)
        rtsp_reader.start()
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._frame_capture_thread, daemon=True)
        self.capture_thread.start()
        
    def _frame_capture_thread(self):
        """Continuously captures frames and adds them to recent frames buffer"""
        logging.info("Frame capture thread started")
        last_frame_time = 0
        
        while not shutdown_event.is_set():
            try:
                # Get frame with timestamp
                frame, timestamp = rtsp_reader.get_frame_with_timestamp()
                
                if frame is not None and isinstance(frame, np.ndarray):
                    if timestamp > last_frame_time:
                        # Add to recent frames buffer
                        frame_buffer.add_recent_frame(frame, timestamp)
                        last_frame_time = timestamp
                time.sleep(0.001)  # Small sleep to prevent CPU overload
            except Exception as e:
                logging.error(f"Error in frame capture thread: {e}", exc_info=True)
                time.sleep(0.1)
    
    def handle_mqtt_event(self, event: Dict[str, Any], original_snapshot=None):
        """
        Handle incoming MQTT detection events
        This is called from the MQTT callback in DetectionLogProcessor
        """
        start_time = time.time()
        
        try:
            # Extract event time
            event_time = event.get('first_seen')
            if isinstance(event_time, str):
                try:
                    event_time = float(event_time)
                except ValueError:
                    event_time = time.time()
            elif event_time is None:
                event_time = time.time()
            
            # Find the best matching frame from recent frames buffer
            best_frame, frame_timestamp = frame_buffer.get_best_frame_for_event(event_time)
            
            # If no good frame found, use the provided snapshot or get current frame
            if best_frame is None:
                if original_snapshot is not None and isinstance(original_snapshot, Image.Image):
                    # Convert PIL Image to OpenCV format
                    best_frame = np.array(original_snapshot.convert('RGB'))
                    best_frame = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
                    frame_timestamp = time.time()
                else:
                    # Get current frame as fallback
                    best_frame = rtsp_reader.get_frame()
                    frame_timestamp = time.time()
                    
                if best_frame is None:
                    logging.error("Failed to get a valid frame for event")
                    return
            
            # Draw bounding box on frame for visualization and save it
            display_frame = best_frame.copy()
            bbox = event.get('bbox')
            cropped_frame = None
            
            if bbox and len(bbox) == 4:
                try:
                    x_center_norm, y_center_norm, width_norm, height_norm = bbox
                    frame_height, frame_width = display_frame.shape[:2]
                    x_center = x_center_norm * frame_width
                    y_center = y_center_norm * frame_height
                    width = width_norm * frame_width
                    height = height_norm * frame_height
                    x1 = max(0, int(x_center - width / 2))
                    y1 = max(0, int(y_center - height / 2))
                    x2 = min(frame_width - 1, int(x_center + width / 2))
                    y2 = min(frame_height - 1, int(y_center + height / 2))
                    
                    # Check that bounding box dimensions are valid
                    if x2 > x1 and y2 > y1:
                        # Draw bounding box on display frame
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        event_type = event.get('event', 'unknown')
                        object_id = event.get('object_id', 'unknown')
                        label = f"{event_type.upper()}: Object {object_id}"
                        cv2.putText(display_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Crop the region of interest from the original frame
                        cropped_frame = best_frame[y1:y2, x1:x2].copy()
                    else:
                        logging.warning(f"Invalid bounding box: ({x1},{y1}), ({x2},{y2})")
                        cropped_frame = best_frame.copy()
                except Exception as e:
                    logging.error(f"Error processing bounding box: {e}", exc_info=True)
                    cropped_frame = best_frame.copy()
            else:
                cropped_frame = best_frame.copy()
            
            # Save the visualization frame
            try:
                event_type = event.get('event', 'unknown')
                object_id = event.get('object_id', 'unknown')
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(config["output_dir"], 
                                          f"{event_type}_obj{object_id}_{timestamp_str}.jpg")
                cv2.imwrite(output_path, display_frame)
                logging.info(f"Saved event frame to {output_path}")
            except Exception as e:
                logging.error(f"Error saving event frame: {e}", exc_info=True)
            
            # Put the cropped frame in the processing buffer
            if cropped_frame is not None:
                frame_buffer.put(cropped_frame, event, frame_timestamp)
                logging.info(f"Queued frame for processing: obj={event.get('object_id', 'unknown')}")
            
            # Record performance metrics
            process_time = time.time() - start_time
            update_metrics(acquisition_times, process_time)
            if process_time > 0.1:
                logging.warning(f"Slow event processing in acquisition pipeline: {process_time:.3f}s")
                
        except Exception as e:
            logging.error(f"Error in handle_mqtt_event: {e}", exc_info=True)

# Pipeline 2: Plate Detection and OCR
class PlateProcessingPipeline:
    """
    Pipeline 2: Processes frames from the buffer for license plate detection and OCR
    """
    def __init__(self, plate_model_path, ocr_model_path):
        self.plate_detector = PlateDetector(plate_model_path)
        self.ocr_plate = OCRPlate(ocr_model_path)
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()
        logging.info("Plate processing pipeline initialized")
    
    def _processing_thread(self):
        """Main processing loop that pulls frames from buffer and processes them"""
        consecutive_errors = 0
        
        while not shutdown_event.is_set():
            try:
                # Get next frame packet from buffer
                packet = frame_buffer.get(timeout=0.5)
                if packet is None:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                frame_age = time.time() - packet.timestamp
                
                # Skip if frame is too old (more than 2 seconds)
                if frame_age > 2.0:
                    logging.warning(f"Skipping old frame in processing pipeline: {frame_age:.2f}s old")
                    frame_buffer.task_done()
                    continue
                
                # Convert to PIL Image for plate detector
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
                    
                    # Detect license plate
                    plate_region = self.plate_detector.detect_plate(pil_image)
                    
                    if plate_region:
                        # Perform OCR on detected plate
                        plate_text, province = self.ocr_plate.predict(plate_region)
                        logging.info(f"OCR Result: Plate: {plate_text}, Province: {province}")
                        
                        # Save the plate image
                        event = packet.event
                        if "object_id" in event:
                            object_id = event["object_id"]
                            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                            plate_path = os.path.join(config["output_dir"], 
                                                     f"plate_{object_id}_{timestamp_str}.jpg")
                            plate_region.save(plate_path)
                            logging.info(f"Saved plate image to {plate_path}")
                            
                            # Create result record
                            result = {
                                "object_id": object_id,
                                "event_type": event.get("event", "unknown"),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "plate_text": plate_text,
                                "province": province,
                                "frame_age": frame_age,
                                "processing_time": time.time() - start_time
                            }
                            
                            # Save result
                            with file_lock:
                                processed_results.append(result)
                                with open(os.path.join(config["output_dir"], "plates_results.json"), "w",encoding="utf-8") as f:
                                    json.dump(processed_results, f,ensure_ascii=False, indent=2)
                    else:
                        logging.info("No license plate detected")
                
                except Exception as e:
                    logging.error(f"Error processing frame: {e}", exc_info=True)
                    consecutive_errors += 1
                finally:
                    # Mark task as done
                    frame_buffer.task_done()
                    
                    # Record performance metrics
                    process_time = time.time() - start_time
                    update_metrics(processing_times, process_time)
                    if process_time > 0.5:
                        logging.warning(f"Slow processing in plate pipeline: {process_time:.3f}s")
                    
                    consecutive_errors = 0
            
            except Exception as e:
                logging.error(f"Error in processing thread: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors > 5:
                    time.sleep(1)  # Back off if multiple errors

def mqtt_callback(original_snapshot, event):
    """
    Callback function for DetectionLogProcessor
    Forwards events to the acquisition pipeline
    """
    global acquisition_pipeline
    acquisition_pipeline.handle_mqtt_event(event, original_snapshot)

def metrics_reporter():
    """Thread that periodically logs performance metrics"""
    while not shutdown_event.is_set():
        log_metrics()
        frame_buffer.log_stats()
        time.sleep(30)  # Report every 30 seconds

def main():
    """Main entry point"""
    global acquisition_pipeline, processing_pipeline, detection_processor
    
    try:
        # Set process priority if possible
        try:
            import psutil
            process = psutil.Process()
            process.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -10)
            logging.info("Set process to high priority")
        except (ImportError, PermissionError, AttributeError):
            logging.warning("Could not set process priority")
        
        # Initialize pipeline 1: Frame acquisition
        logging.info("Initializing frame acquisition pipeline")
        rtsp_url = config.get('rtsp_url')
        if not rtsp_url:
            raise ValueError("RTSP URL missing from config")
        acquisition_pipeline = FrameAcquisitionPipeline(rtsp_url)
        
        # Initialize pipeline 2: Plate processing
        logging.info("Initializing plate processing pipeline")
        plate_model_path = config.get('plate_detector_model_path')
        ocr_model_path = config.get('ocr_plate_model_path')
        processing_pipeline = PlateProcessingPipeline(plate_model_path, ocr_model_path)
        
        # Start metrics reporter
        metrics_thread = threading.Thread(target=metrics_reporter, daemon=True)
        metrics_thread.start()
        
        # Initialize and start detection processor
        logging.info("Initializing MQTT detection processor")
        detection_processor = DetectionLogProcessor("config.json", mqtt_callback)
        
        # Main loop
        logging.info("Dual pipeline system running")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Clean shutdown
        shutdown_event.set()
        if detection_processor:
            detection_processor.client.disconnect()
        if rtsp_reader:
            rtsp_reader.stop()
        logging.info("System shutdown complete")

if __name__ == "__main__":
    main()