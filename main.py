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
import platform
import psutil

from detection_log_processor import DetectionLogProcessor
from plate_detector import PlateDetector
from ocr_plate import OCRPlate
from frame_buffer import FrameBuffer

# Configure logging with microsecond precision for timing analysis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)

# --- Load configuration from config.json ---
with open("config.json", "r") as f:
    config = json.load(f)

os.makedirs(config["output_dir"], exist_ok=True)

# Global logs and locks
detection_log_list = []
processed_results = []
file_lock = threading.RLock()

# Performance metrics
acquisition_times = []
processing_times = []
METRICS_MAX_SAMPLES = 100
metrics_lock = threading.RLock()

# Shutdown event for clean termination
shutdown_event = threading.Event()

# Create the shared frame buffer between pipelines
frame_buffer = FrameBuffer(max_size=config.get("buffer_size", 30))

# Global reference to components
rtsp_reader = None
detection_processor = None
acquisition_pipeline = None
processing_pipeline = None

# Set process and thread priorities
def set_high_priority():
    """Set the current process to high priority"""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 
                15  # THREAD_PRIORITY_TIME_CRITICAL
            )
        elif system == "Linux":
            os.nice(-19)  # Requires root privileges
        
        # Also try psutil which works on more platforms
        try:
            process = psutil.Process()
            if system == "Windows":
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            elif system in ("Linux", "Darwin"):
                process.nice(-19)
        except:
            pass
            
        logging.info(f"Set high priority for thread {threading.current_thread().name}")
        return True
    except Exception as e:
        logging.warning(f"Failed to set high thread priority: {e}")
        return False

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
            min_acquisition = min(acquisition_times)
            logging.info(f"Acquisition Pipeline: Avg={avg_acquisition*1000:.1f}ms, "
                         f"Min={min_acquisition*1000:.1f}ms, Max={max_acquisition*1000:.1f}ms")
        
        if processing_times:
            avg_processing = sum(processing_times) / len(processing_times)
            max_processing = max(processing_times)
            min_processing = min(processing_times)
            logging.info(f"Processing Pipeline: Avg={avg_processing*1000:.1f}ms, "
                         f"Min={min_processing*1000:.1f}ms, Max={max_processing*1000:.1f}ms")

# Pipeline 1: Frame Acquisition
class FrameAcquisitionPipeline:
    """
    Pipeline 1: Ultra-low latency frame capture from RTSP with optimized event handling
    """
    def __init__(self, rtsp_url):
        global rtsp_reader
        
        # Initialize RTSP reader
        from rtsp_frame_reader import RTSPFrameReader
        rtsp_reader = RTSPFrameReader(rtsp_url)
        rtsp_reader.start()
        
        # Start frame capture thread with high priority
        self.capture_thread = threading.Thread(
            target=self._frame_capture_thread, 
            daemon=True,
            name="Frame-Capture"
        )
        self.capture_thread.start()
        
    def _frame_capture_thread(self):
        """Continuously captures frames with high priority"""
        logging.info("Frame capture thread started")
        set_high_priority()  # Make this thread high priority
        
        last_frame_time = 0
        
        while not shutdown_event.is_set():
            try:
                # Get the best quality frame with timestamp
                frame, timestamp = rtsp_reader.get_best_frame()
                
                if frame is not None and isinstance(frame, np.ndarray):
                    if timestamp > last_frame_time:
                        # Add to recent frames buffer
                        frame_buffer.add_recent_frame(frame, timestamp)
                        last_frame_time = timestamp
                
                # Adaptive sleep to match frame rate
                time.sleep(0.001)  # Minimal sleep to prevent CPU overload
                
            except Exception as e:
                logging.error(f"Error in frame capture thread: {e}", exc_info=True)
                time.sleep(0.1)
    
    def handle_mqtt_event(self, event: Dict[str, Any], original_snapshot=None):
        """
        High-performance event handler that finds the optimal frame for each detection event
        """
        start_time = time.time()
        
        try:
            # Extract event info with robust error handling
            event_time = event.get('first_seen')
            if isinstance(event_time, str):
                try:
                    event_time = float(event_time)
                except ValueError:
                    event_time = time.time()
            elif event_time is None:
                event_time = time.time()
            
            # Find the best matching frame from advanced buffer
            best_frame, frame_timestamp = frame_buffer.get_best_frame_for_event(event)
            
            # If no good frame found, use the provided snapshot or get current frame
            if best_frame is None:
                if original_snapshot is not None and isinstance(original_snapshot, Image.Image):
                    # Convert PIL Image to OpenCV format
                    best_frame = np.array(original_snapshot.convert('RGB'))
                    best_frame = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
                    frame_timestamp = time.time()
                else:
                    # Get best available frame as fallback
                    best_frame, frame_timestamp = rtsp_reader.get_best_frame()
                    
                    if best_frame is None:
                        # Last resort: get current frame
                        best_frame = rtsp_reader.get_frame()
                        frame_timestamp = time.time()
                    
                if best_frame is None:
                    logging.error("Failed to get a valid frame for event")
                    return
                    
            # Check frame age - we want fresh frames
            frame_age = time.time() - frame_timestamp
            if frame_age > 1.0:  # Over 1 second old
                logging.warning(f"Using older frame ({frame_age:.2f}s old) for processing")
            
            # Draw bounding box on frame for visualization and save it
            display_frame = best_frame.copy()
            bbox = event.get('bbox')
            cropped_frame = None
            
            if bbox and len(bbox) == 4:
                try:
                    x_center_norm, y_center_norm, width_norm, height_norm = bbox
                    frame_height, frame_width = display_frame.shape[:2]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_center = x_center_norm * frame_width
                    y_center = y_center_norm * frame_height
                    width = width_norm * frame_width
                    height = height_norm * frame_height
                    
                    # Calculate bounding box coordinates
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
                        
                        # Add frame age to the display
                        cv2.putText(display_frame, 
                                   f"Frame age: {frame_age*1000:.0f}ms", 
                                   (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Crop the region of interest with margins for better plate detection
                        # Add 10% margin to each side of bounding box for better context
                        margin_x = int(width * 0.1)
                        margin_y = int(height * 0.1)
                        crop_x1 = max(0, x1 - margin_x)
                        crop_y1 = max(0, y1 - margin_y)
                        crop_x2 = min(frame_width - 1, x2 + margin_x)
                        crop_y2 = min(frame_height - 1, y2 + margin_y)
                        
                        cropped_frame = best_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
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
                logging.info(f"Saved event frame to {output_path} (age: {frame_age*1000:.0f}ms)")
            except Exception as e:
                logging.error(f"Error saving event frame: {e}", exc_info=True)
            
            # Put the cropped frame in the processing buffer with high priority
            if cropped_frame is not None:
                frame_buffer.put(cropped_frame, event, frame_timestamp)
                logging.info(f"Queued frame for processing: obj={event.get('object_id', 'unknown')}")
            
            # Record performance metrics
            process_time = time.time() - start_time
            update_metrics(acquisition_times, process_time)
            if process_time > 0.05:  # More than 50ms is slow for acquisition
                logging.warning(f"Slow event processing in acquisition: {process_time*1000:.1f}ms")
                
        except Exception as e:
            logging.error(f"Error in handle_mqtt_event: {e}", exc_info=True)

# Pipeline 2: Plate Detection and OCR with multi-threading
class PlateProcessingPipeline:
    """
    Pipeline 2: Multi-threaded plate detection and OCR for high throughput
    """
    def __init__(self, plate_model_path, ocr_model_path, num_workers=1):
        # Initialize models (shared across threads)
        self.plate_detector = PlateDetector(plate_model_path)
        self.ocr_plate = OCRPlate(ocr_model_path)
        
        # Start processing workers
        self.num_workers = num_workers
        self.worker_threads = []
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._processing_worker,
                daemon=True,
                name=f"Plate-Worker-{i+1}"
            )
            thread.start()
            self.worker_threads.append(thread)
            
        logging.info(f"Plate processing pipeline initialized with {num_workers} workers")
    
    def _processing_worker(self):
        """Worker thread that processes frames from the buffer"""
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
                    logging.warning(f"Skipping old frame in processing: {frame_age:.2f}s old")
                    frame_buffer.task_done()
                    continue
                
                # Convert to PIL Image for plate detector
                try:
                    # First check if the frame is valid
                    if packet.frame is None or not isinstance(packet.frame, np.ndarray):
                        logging.error("Invalid frame in packet")
                        frame_buffer.task_done()
                        continue
                        
                    # Convert to RGB for PIL
                    pil_image = Image.fromarray(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
                    
                    # Detect license plate with timing
                    plate_start = time.time()
                    plate_region = self.plate_detector.detect_plate(pil_image)
                    plate_time = time.time() - plate_start
                    
                    if plate_region:
                        # Perform OCR on detected plate
                        ocr_start = time.time()
                        plate_text, province = self.ocr_plate.predict(plate_region)
                        ocr_time = time.time() - ocr_start
                        
                        logging.info(f"OCR Result: Plate: {plate_text}, Province: {province} "
                                    f"(detect: {plate_time*1000:.1f}ms, OCR: {ocr_time*1000:.1f}ms)")
                        
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
                                "processing_time": time.time() - start_time,
                                "detection_time": plate_time,
                                "ocr_time": ocr_time
                            }
                            
                            # Save result atomically
                            with file_lock:
                                processed_results.append(result)
                                with open(os.path.join(config["output_dir"], "plates_results.json"), "w") as f:
                                    json.dump(processed_results, f, indent=2)
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
                        logging.warning(f"Slow processing in plate pipeline: {process_time*1000:.1f}ms")
                    
                    # Reset error counter if successful
                    if consecutive_errors > 0:
                        consecutive_errors -= 1
            
            except Exception as e:
                logging.error(f"Error in processing thread: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors > 5:
                    # Back off if persistent errors
                    time.sleep(1)

def mqtt_callback(original_snapshot, event):
    """
    Optimized callback function for DetectionLogProcessor
    Forwards events to the acquisition pipeline
    """
    global acquisition_pipeline
    
    # Add event timestamp for tracking latency
    event['_received_time'] = time.time()
    
    # Pass to acquisition pipeline
    acquisition_pipeline.handle_mqtt_event(event, original_snapshot)

def metrics_reporter():
    """Thread that periodically logs performance metrics"""
    while not shutdown_event.is_set():
        log_metrics()
        frame_buffer.log_stats()
        
        # Report system resource usage
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            mem_info = process.memory_info()
            logging.info(f"System resources: CPU={cpu_percent:.1f}%, "
                         f"Memory={mem_info.rss/(1024*1024):.1f}MB")
        except Exception:
            pass
            
        # Report maximum latency
        try:
            if processed_results:
                recent_results = processed_results[-10:]
                times = [r.get('frame_age', 0) for r in recent_results]
                if times:
                    avg_age = sum(times) / len(times)
                    logging.info(f"Average frame age at processing: {avg_age*1000:.1f}ms")
        except Exception:
            pass
            
        time.sleep(30)  # Report every 30 seconds

def adaptive_performance_tuner():
    """
    Thread that monitors system performance and dynamically adjusts parameters
    to maintain optimal performance
    """
    global frame_buffer
    
    logging.info("Starting adaptive performance tuning")
    
    while not shutdown_event.is_set():
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Adjust parameters based on CPU usage
            if cpu_percent > 90:  # High CPU load
                # Make frame selection more aggressive to reduce workload
                logging.info(f"High CPU load ({cpu_percent}%), adjusting parameters")
                # Future logic to adjust parameters
                
            elif cpu_percent < 50:  # Low CPU load
                # Allow more processing to improve quality
                pass
                
            # Check processing times
            with metrics_lock:
                if processing_times and len(processing_times) > 10:
                    avg_time = sum(processing_times[-10:]) / 10
                    if avg_time > 0.5:  # Processing taking > 500ms
                        logging.info(f"Slow processing detected ({avg_time*1000:.1f}ms), adjusting parameters")
                        # Future logic to adjust parameters
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logging.error(f"Error in adaptive tuner: {e}")
            time.sleep(10)

def main():
    """Main entry point with improved thread management and error handling"""
    global acquisition_pipeline, processing_pipeline, detection_processor
    
    try:
        # Set main process to high priority
        set_high_priority()
        
        logging.info("Starting License Plate Recognition System...")
        
        # Initialize Pipeline 1: Frame acquisition
        logging.info("Initializing frame acquisition pipeline")
        rtsp_url = config.get('rtsp_url')
        if not rtsp_url:
            raise ValueError("RTSP URL missing from config")
        acquisition_pipeline = FrameAcquisitionPipeline(rtsp_url)
        
        # Initialize Pipeline 2: Plate processing
        logging.info("Initializing plate processing pipeline")
        plate_model_path = config.get('plate_detector_model_path')
        ocr_model_path = config.get('ocr_plate_model_path')
        
        # Determine optimal number of worker threads based on CPU cores
        worker_threads = max(1, min(2, psutil.cpu_count(logical=False) - 1))
        logging.info(f"Using {worker_threads} worker threads for processing")
        
        processing_pipeline = PlateProcessingPipeline(
            plate_model_path, 
            ocr_model_path,
            num_workers=worker_threads
        )
        
        # Start metrics reporter
        metrics_thread = threading.Thread(
            target=metrics_reporter, 
            daemon=True,
            name="Metrics-Reporter"
        )
        metrics_thread.start()
        
        # Start adaptive performance tuner
        tuner_thread = threading.Thread(
            target=adaptive_performance_tuner,
            daemon=True,
            name="Performance-Tuner"
        )
        tuner_thread.start()
        
        # Initialize and start detection processor
        logging.info("Initializing MQTT detection processor")
        detection_processor = DetectionLogProcessor("config.json", mqtt_callback)
        
        # Main loop
        logging.info("Real-time license plate recognition system running")
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
            try:
                detection_processor.client.disconnect()
            except:
                pass
        if rtsp_reader:
            try:
                rtsp_reader.stop()
            except:
                pass
        logging.info("System shutdown complete")

if __name__ == "__main__":
    main()