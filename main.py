from datetime import datetime
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
import concurrent.futures

from tools.detection_log_processor import DetectionLogProcessor
from tools.plate_detector import PlateDetector
from tools.ocr_plate import OCRPlate
from tools.frame_buffer import TimeSyncedFrameBuffer
from tools.ocr_processor_pool import OCRProcessorPool

# Configure logging with microsecond precision for timing analysis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)

# --- Load configuration from config.json ---
with open("config/config.json", "r") as f:
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

# Detection log storage
detection_logs = []
detection_log_lock = threading.RLock()
detection_log_last_save_time = time.time()
detection_log_save_interval = 30  # Seconds between saves

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
    Optimized plate detection and OCR pipeline with dedicated OCR thread pool
    """
    def __init__(self, plate_model_path, ocr_model_path, num_workers=1):
        # Initialize models
        self.plate_detector = PlateDetector(plate_model_path)
        self.ocr_plate = OCRPlate(ocr_model_path)
        
        # Create OCR processor pool for handling the OCR bottleneck
        ocr_threads = max(1, min(3, psutil.cpu_count(logical=False)))
        self.ocr_pool = OCRProcessorPool(num_workers=ocr_threads, ocr_instance=self.ocr_plate)
        
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
            
        logging.info(f"Plate processing pipeline initialized with {num_workers} workers and {ocr_threads} OCR threads")
    
    def _ocr_result_callback(self, plate_number, province, object_id, event_data):
        """Handle OCR results from the dedicated OCR pool and save to detection logs"""
        if not plate_number or not province:
            logging.warning(f"OCR failed for object {object_id}")
            
            # Log failed OCR attempts too
            with detection_log_lock:
                detection_log = {
                    "object_id": object_id,
                    "timestamp": time.time(),
                    "event_time": event_data.get('event_time'),
                    "ocr_success": False,
                    "event_type": event_data.get('original_event', {}).get('event', 'unknown'),
                    "detection_confidence": event_data.get('confidence', 0),
                    "processing_latency": time.time() - event_data.get('processing_start', time.time()),
                    "ocr_status": "failed"
                }
                detection_logs.append(detection_log)
            
            return
            
        # Process successful OCR result
        try:
            # Get timing data from the event
            event_time = event_data.get('event_time')
            detection_time = event_data.get('detection_time', 0)
            
            # Calculate total processing time
            processing_start = event_data.get('processing_start', 0)
            total_time = time.time() - processing_start if processing_start else 0
            ocr_time = total_time - detection_time if detection_time else 0
            
            # Create result record
            result = {
                "object_id": object_id,
                "event_time": event_time,
                "processing_time": time.time(),
                "plate_number": plate_number,
                "province": province,
                "confidence": event_data.get('confidence', 0),
                "processing_latency": total_time,
                "detection_time": detection_time,
                "ocr_time": ocr_time,
                "event": event_data.get('original_event', {}).copy()
            }
            
            # Calculate end-to-end latency
            if event_time:
                result["event_to_detection_latency"] = time.time() - event_time
            
            # Save results
            with file_lock:
                processed_results.append(result)
                
                # Save to file periodically
                if len(processed_results) % 5 == 0:
                    output_file = os.path.join(config["output_dir"], "results.json")
                    try:
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(processed_results, f, ensure_ascii=False)
                    except Exception as e:
                        logging.error(f"Error saving results: {e}")
            
            # Add to detection logs with OCR output
            with detection_log_lock:
                detection_log = {
                    "object_id": object_id,
                    "timestamp": time.time(),
                    "event_time": event_time,
                    "ocr_success": True,
                    "plate_number": plate_number,
                    "province": province,
                    "detection_confidence": event_data.get('confidence', 0),
                    "processing_latency": total_time,
                    "detection_time": detection_time,
                    "ocr_time": ocr_time,
                    "event_type": event_data.get('original_event', {}).get('event', 'unknown'),
                    "ocr_status": "success",
                    "frame_timestamp": event_data.get('timestamp'),
                    "rtsp_frame_count": event_data.get('original_event', {}).get('frame_count', 0)
                }
                detection_logs.append(detection_log)
                
                # Check if we should save logs
                if time.time() - detection_log_last_save_time > detection_log_save_interval:
                    save_detection_logs()
            
            # Log results
            logging.info(f"Successfully processed: {plate_number} {province} | "
                      f"Obj: {object_id} | "
                      f"Time: {total_time*1000:.1f}ms ({detection_time*1000:.1f}ms detect, "
                      f"{ocr_time*1000:.1f}ms OCR)")
            
            # Record metrics for monitoring
            update_metrics(processing_times, total_time)
            
            # Save plate image for verification if available
            plate_img = event_data.get('plate_img')
            if plate_img:
                try:
                    timestamp_str = datetime.fromtimestamp(event_data.get('timestamp', time.time())).strftime('%Y%m%d_%H%M%S')
                    filename = f"plate_{plate_number}_{object_id}_{timestamp_str}.jpg"
                    save_path = os.path.join(config["output_dir"], filename)
                    plate_img.save(save_path)
                except Exception as e:
                    logging.error(f"Error saving plate image: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing OCR result: {e}", exc_info=True)
    
    def _processing_worker(self):
        """Worker thread that processes frames with parallel OCR handling"""
        consecutive_errors = 0
        consecutive_empties = 0
        
        # For batch processing when multiple detections arrive
        batch_size = 3
        detection_batch = []
        last_batch_time = time.time()
        
        # Create ThreadPoolExecutor for parallel OCR
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            while not shutdown_event.is_set():
                try:
                    # Get next frame packet from buffer
                    packet = frame_buffer.get(timeout=0.5)
                    if packet is None:
                        # Process any remaining batch items if we've waited too long
                        if detection_batch and time.time() - last_batch_time > 0.5:
                            self._process_detection_batch(detection_batch, executor)
                            detection_batch = []
                            last_batch_time = time.time()
                            
                        consecutive_empties += 1
                        if consecutive_empties > 5:
                            time.sleep(0.1)
                        continue
                        
                    consecutive_empties = 0
                    consecutive_errors = 0
                    
                    # Process frame detection as before...
                    frame, event, timestamp = packet
                    start_time = time.time()
                    object_id = event.get('object_id', 'unknown')
                    
                    # Update adaptive parameters
                    backlog_count = frame_buffer.get_stats()['unprocessed']
                    self.plate_detector.set_backlog(backlog_count)
                    self.ocr_plate.set_backlog(backlog_count)
                    
                    # Skip old frames during high load
                    frame_age = time.time() - timestamp
                    if backlog_count > 10 and frame_age > 2.0:
                        logging.warning(f"Skipping old frame ({frame_age:.1f}s) due to backlog ({backlog_count})")
                        continue
                    
                    # Convert to PIL format for detection
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Detect license plate
                    plate_img = self.plate_detector.detect_plate(pil_image, object_id, event)
                    detection_time = time.time() - start_time
                    
                    if plate_img is not None:
                        # New: Add detection to batch
                        detection_data = {
                            'plate_img': plate_img,
                            'object_id': object_id,
                            'event_time': event.get('first_seen', timestamp),
                            'timestamp': timestamp, 
                            'confidence': event.get('confidence', 0),
                            'detection_time': detection_time,
                            'processing_start': start_time,
                            'original_event': event.copy()
                        }
                        
                        detection_batch.append(detection_data)
                        logging.info(f"License plate detected for obj {object_id} in {detection_time*1000:.1f}ms, queued for OCR")
                        
                        # Process batch when it reaches size or when high priority
                        if len(detection_batch) >= batch_size or event.get('confidence', 0) > 0.8:
                            self._process_detection_batch(detection_batch, executor)
                            detection_batch = []
                            last_batch_time = time.time()
                except Exception as e:
                    logging.error(f"Error in processing worker: {e}", exc_info=True)
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        time.sleep(1.0)
                        consecutive_errors = 0
    
    def _process_detection_batch(self, batch, executor):
        """Process a batch of detections with parallel OCR"""
        if not batch:
            return
            
        # Submit all OCR tasks to the executor
        future_to_detection = {
            executor.submit(
                self.ocr_plate.predict, 
                detection['plate_img'], 
                detection['object_id']
            ): detection for detection in batch
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_detection):
            detection = future_to_detection[future]
            try:
                plate_number, province = future.result()
                
                # FIX: Properly validate both plate number and province before callback
                if plate_number and province and plate_number != "Unknown":
                    # Create a result record
                    event_data = detection
                    self._ocr_result_callback(plate_number, province, detection['object_id'], event_data)
                else:
                    object_id = detection.get('object_id', 'unknown')
                    # Give more detailed log about what's missing
                    if not plate_number:
                        logging.warning(f"OCR failed - no plate number for object {object_id}")
                    elif plate_number == "Unknown":
                        logging.warning(f"OCR failed - unknown plate for object {object_id}")
                    elif not province:
                        logging.warning(f"OCR failed - no province for object {object_id}")
                    elif province == "Unknown":
                        # Province unknown but plate OK is acceptable
                        event_data = detection
                        self._ocr_result_callback(plate_number, "Unknown", object_id, event_data)
                        logging.info(f"OCR partially successful - plate {plate_number} without province")
                        
            except Exception as e:
                logging.error(f"Error in OCR task: {e}", exc_info=True)

# Update the metrics_reporter function

def metrics_reporter():
    """Thread that periodically logs performance metrics and saves detection logs"""
    while not shutdown_event.is_set():
        log_metrics()
        frame_buffer.log_stats()
        
        # Save detection logs periodically
        with detection_log_lock:
            if detection_logs and time.time() - detection_log_last_save_time > detection_log_save_interval:
                save_detection_logs()
                
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

from tools.performance_monitor import PerformanceMonitor

# Initialize performance monitor
performance_monitor = None

def initialize_performance_monitor():
    """Initialize and configure the performance monitoring system"""
    global performance_monitor
    
    # Create monitor instance
    performance_monitor = PerformanceMonitor(config_path="config/config.json")
    
    # Register system components
    if frame_buffer and processing_pipeline:
        performance_monitor.register_components(
            frame_buffer=frame_buffer,
            ocr_pool=processing_pipeline.ocr_pool,
            plate_detector=processing_pipeline.plate_detector
        )
        
    logging.info("Performance monitoring system initialized")
    return performance_monitor

# Function to save detection logs to JSON file
def save_detection_logs():
    """Save accumulated detection logs to JSON file"""
    global detection_log_last_save_time
    
    with detection_log_lock:
        if not detection_logs:
            return
            
        try:
            output_file = os.path.join(config["output_dir"], "detection_log.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(detection_logs, f, ensure_ascii=False)
            logging.info(f"Saved {len(detection_logs)} detection logs to {output_file}")
            detection_log_last_save_time = time.time()
        except Exception as e:
            logging.error(f"Error saving detection logs: {e}")

# Add this function after save_detection_logs()

def analyze_ocr_performance():
    """Analyze OCR performance statistics and log summary"""
    with detection_log_lock:
        if not detection_logs:
            return
            
        total = len([log for log in detection_logs if log.get('event_phase') != 'initial_detection'])
        successful = len([log for log in detection_logs if log.get('ocr_success', False)])
        failed = total - successful
        success_rate = (successful / total * 100) if total > 0 else 0
        
        # Calculate average OCR time
        ocr_times = [log.get('ocr_time', 0) for log in detection_logs if 'ocr_time' in log]
        avg_ocr_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0
        
        logging.info(f"OCR Performance: {successful}/{total} successful ({success_rate:.1f}%), "
                    f"Average OCR time: {avg_ocr_time*1000:.1f}ms")
        
        # Log the most common plate numbers
        plate_counts = {}
        for log in detection_logs:
            if log.get('plate_number'):
                plate = log.get('plate_number')
                plate_counts[plate] = plate_counts.get(plate, 0) + 1
                
        if plate_counts:
            top_plates = sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logging.info(f"Most common plates: {', '.join([f'{plate}({count})' for plate, count in top_plates])}")

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
        detection_processor = DetectionLogProcessor("config/config.json", mqtt_callback)
        
        # Initialize performance monitor
        initialize_performance_monitor()
        
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
        
        # Save final detection logs before exiting
        logging.info("Saving final detection logs...")
        save_detection_logs()
        
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