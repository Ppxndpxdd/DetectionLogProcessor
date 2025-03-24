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
from frame_buffer import TimeSyncedFrameBuffer

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
frame_buffer = TimeSyncedFrameBuffer(max_size=config.get("buffer_size", 30))

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

def mqtt_callback(original_snapshot, event):
    """
    High-performance callback for MQTT events
    Uses burst capture and precise frame sync
    """
    start_time = time.time()
    
    try:
        # Record receipt time
        event_time = event.get('first_seen')
        receipt_time = event.get('_received_time', time.time())
        
        if isinstance(event_time, str):
            try:
                event_time = float(event_time)
            except ValueError:
                event_time = receipt_time
        elif event_time is None:
            event_time = receipt_time
            
        # Store event in logs
        detection_log_list.append(event.copy())
        
        # Process the cropped image for plate detection
        if original_snapshot is not None:
            # Convert to OpenCV format if needed
            if isinstance(original_snapshot, Image.Image):
                frame = np.array(original_snapshot.convert('RGB'))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = original_snapshot
                
            # Queue for processing
            frame_buffer.put(frame, event, receipt_time)
            logging.info(f"Queued frame for processing: obj={event.get('object_id', 'unknown')}")
            
        # Record performance
        process_time = time.time() - start_time
        update_metrics(acquisition_times, process_time)
        
    except Exception as e:
        logging.error(f"Error in MQTT callback: {e}", exc_info=True)

# Pipeline for plate processing
class PlateProcessingPipeline:
    """
    Optimized plate detection and OCR pipeline
    """
    def __init__(self, plate_model_path, ocr_model_path, num_workers=1):
        # Initialize models
        self.plate_detector = PlateDetector(plate_model_path)
        self.ocr_plate = OCRPlate(ocr_model_path)
        
        # Start worker threads
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
                
                # Skip if frame is too old
                if frame_age > 2.0:
                    logging.warning(f"Skipping old frame in processing: {frame_age:.2f}s old")
                    frame_buffer.task_done()
                    continue
                
                # Process the frame
                try:
                    # Convert to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
                    
                    # Detect license plate
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
                                "ocr_time": ocr_time,
                                "event_to_detection_latency": time.time() - packet.timestamp
                            }
                            
                            # Save result
                            with file_lock:
                                processed_results.append(result)
                                with open(os.path.join(config["output_dir"], "plates_results.json"), "w", encoding="utf-8") as f:
                                    json.dump(processed_results, f, ensure_ascii=False, indent=2)
                    else:
                        logging.info("No license plate detected")
                
                except Exception as e:
                    logging.error(f"Error processing plate: {e}", exc_info=True)
                    consecutive_errors += 1
                finally:
                    # Mark task as done
                    frame_buffer.task_done()
                    
                    # Record metrics
                    process_time = time.time() - start_time
                    update_metrics(processing_times, process_time)
                    
                    # Reset error counter on success
                    consecutive_errors = max(0, consecutive_errors - 1)
                    
            except Exception as e:
                logging.error(f"Error in processing worker: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors > 5:
                    time.sleep(1)  # Back off on persistent errors

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
            
        # Report latency metrics
        try:
            if processed_results:
                recent_results = processed_results[-10:]
                e2e_latencies = [r.get('event_to_detection_latency', 0) for r in recent_results if 'event_to_detection_latency' in r]
                if e2e_latencies:
                    avg_latency = sum(e2e_latencies) / len(e2e_latencies)
                    max_latency = max(e2e_latencies)
                    logging.info(f"End-to-end latency: Avg={avg_latency*1000:.1f}ms, Max={max_latency*1000:.1f}ms")
        except Exception as e:
            logging.error(f"Error reporting metrics: {e}")
            
        time.sleep(30)  # Report every 30 seconds

def main():
    """Main entry point"""
    global detection_processor, processing_pipeline
    
    try:
        # Set main process to high priority
        set_high_priority()
        
        logging.info("Starting Enhanced License Plate Recognition System...")
        
        # Initialize plate processing pipeline
        logging.info("Initializing plate processing pipeline")
        plate_model_path = config.get('plate_detector_model_path')
        ocr_model_path = config.get('ocr_plate_model_path')
        
        # Determine optimal number of worker threads
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
        
        # Initialize detection processor with advanced frame synchronization
        logging.info("Initializing detection processor with real-time frame synchronization")
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