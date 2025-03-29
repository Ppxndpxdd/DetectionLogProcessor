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
import database_connect.postgres as db
import generate_config.generate_config as config

from minio import Minio
from minio.error import S3Error

from tools.detection_log_processor import DetectionLogProcessor
from tools.plate_detector import PlateDetector
from tools.ocr_plate import OCRPlate
from tools.frame_buffer import TimeSyncedFrameBuffer
from tools.ocr_processor_pool import OCRProcessorPool

def set_variable_from_config():
    # Global variables for all configuration settings
    global broker, port, topic, client_id, username, password, ca_cert
    global plate_weight, ocr_weight
    global minio_endpoint, minio_access_key, minio_secret_key
    global output_dir, buffer_size, performance_monitor_config_path
    global default_rtsp_url  # Add default RTSP URL
    
    # MQTT configuration
    config_data = config.read_mqtt_config("config/subscribe_incident_config.ini")
    broker = config_data['broker']
    port = config_data['port']
    topic = config_data['topic']
    client_id = config_data['client_id']
    username = config_data['username']
    password = config_data['password']
    ca_cert = config_data['ca_certs_path']
    
    # Model weights configuration
    config_data = config.read_model_weight_config("config/model_weight_config.ini")
    output_dir = config_data['output_dir']
    buffer_size = int(config_data.get('buffer_size', 30))
    plate_weight = config_data['plate_weight_path']
    ocr_weight = config_data['ocr_weight_path']
    default_rtsp_url = config_data.get('default_rtsp_url', 'rtsp://161.246.5.10:9999/samui_front')  # Get default RTSP URL
    
    # Minio configuration
    config_data = config.read_minio_config("config/minio_config.ini")
    minio_endpoint = config_data['endpoint']
    minio_access_key = config_data['access_key']
    minio_secret_key = config_data['secret_key']
    
    # Performance monitor configuration path
    performance_monitor_config_path = "config/config.json"

# Configure logging with microsecond precision for timing analysis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)

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

# Global variables for configuration settings
broker = None
port = None
topic = None
client_id = None
username = None
password = None
ca_cert = None
plate_weight = None
ocr_weight = None
minio_endpoint = None
minio_access_key = None
minio_secret_key = None
output_dir = None
buffer_size = 30  # Default value
performance_monitor_config_path = None
default_rtsp_url = None  # Default RTSP URL

# Map to store marker_id to rtsp_url mappings
rtsp_url_cache = {}
rtsp_url_cache_lock = threading.RLock()

# Create the shared frame buffer between pipelines
# Will be initialized after loading config
frame_buffer = None

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

# Function to update or get RTSP URL for a marker_id
def get_rtsp_url_for_marker(marker_id):
    """Get RTSP URL for a given marker ID, either from cache or from database"""
    with rtsp_url_cache_lock:
        # Check if we already have this marker's URL cached
        if marker_id in rtsp_url_cache:
            return rtsp_url_cache[marker_id]
        
        # If not, try to get it from the database
        try:
            nvr_link = db.get_nvr_link_by_marker_id(marker_id)
            if nvr_link:
                logging.info(f"Retrieved NVR link for marker {marker_id}: {nvr_link}")
                rtsp_url_cache[marker_id] = nvr_link
                return nvr_link
        except Exception as e:
            logging.error(f"Error retrieving NVR link for marker {marker_id}: {e}")
    
    # If we couldn't get a specific URL, use the default
    return default_rtsp_url

def update_rtsp_reader_for_marker(marker_id):
    """Update the RTSP reader to use the URL for a specific marker"""
    global rtsp_reader
    
    # Get the appropriate RTSP URL
    rtsp_url = get_rtsp_url_for_marker(marker_id)
    
    if not rtsp_url:
        logging.warning(f"No RTSP URL available for marker {marker_id}, using current stream")
        return False
    
    # If we have an RTSP reader and it's using a different URL, update it
    if rtsp_reader and rtsp_reader.rtsp_url != rtsp_url:
        try:
            logging.info(f"Switching RTSP stream to {rtsp_url} for marker {marker_id}")
            rtsp_reader.update_rtsp_url(rtsp_url)
            return True
        except Exception as e:
            logging.error(f"Failed to update RTSP URL: {e}", exc_info=True)
            return False
    return True

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
        
        # Check if we need to update the RTSP URL for this marker
        marker_id = event.get('marker_id')
        if marker_id:
            # Update the RTSP reader to use the right camera stream for this marker
            update_rtsp_reader_for_marker(marker_id)
        
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

class MinioUpload:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def upload_file(self, source_file, destination_file, bucket_name):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Created bucket: {bucket_name}")
            else:
                print(f"Bucket {bucket_name} already exists")

            self.client.fput_object(bucket_name, destination_file, source_file)
            print(f"{source_file} successfully uploaded as {destination_file} to bucket {bucket_name}")
            return True
        except S3Error as exc:
            print("Error occurred:", exc)
            return False

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
        ocr_threads = max(16, min(3, psutil.cpu_count(logical=False)))
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
        # Only convert empty strings to None, but keep None values as None
        if plate_number == "Unknown":
            plate_number = None
        if province == "Unknown":
            province = None
        
        # Only convert empty strings to None, but keep None values as None
        if plate_number == "":
            plate_number = None
        if province == "":
            province = None
        
        output_path = event_data.get('output_image_path')
        if output_path is None and 'original_event' in event_data:
            output_path = event_data.get('original_event', {}).get('output_image_path')
        print(f"output_path is {output_path}")

        incident_id = event_data.get('incident_id')
        if incident_id is None and 'original_event' in event_data:
            incident_id = event_data.get('original_event', {}).get('incident_id')
            
        print(f"incident_id is {incident_id}")
        nvr_link = event_data.get('nvr_link')
        marker_id = event_data.get('original_event', {}).get('marker_id')
        
        # Create result record regardless of OCR success
        result = {
            "object_id": object_id,
            "event_time": event_data.get('event_time'),
            "processing_time": time.time(),
            "plate_number": plate_number if plate_number else None,
            "province": province if province else None,
            "confidence": event_data.get('confidence', 0),
            "image_path": output_path,
            "incident_id": incident_id,
            "nvr_link": nvr_link,
            "marker_id": marker_id
        }
        
        # Process OCR result
        try:
            # Get timing data from the event
            event_time = event_data.get('event_time')
            detection_time = event_data.get('detection_time', 0)
            
            # Calculate total processing time
            processing_start = event_data.get('processing_start', 0)
            total_time = time.time() - processing_start if processing_start else 0
            ocr_time = total_time - detection_time if detection_time else 0
            
            # Log the appropriate message based on OCR success
            if plate_number is not None:
                if province is not None:
                    logging.info(f"OCR successful - plate {plate_number}, province {province}")
                else:
                    logging.info(f"OCR partially successful - plate {plate_number} without province")
            else:
                logging.warning(f"OCR failed for object {object_id}")
            
            # Update the incident record in the database regardless of OCR success
            # Following the exact pattern from incident_subscribe.py
            if incident_id:
                try:
                    # Set image path based on whether plate was found (exact match with incident_subscribe.py)
                    image_path = str(incident_id) + ".jpg" if plate_number is not None else "No image"
                    
                    # Convert None values to "No number" and "No province" for database
                    db_plate_number = "No number" if plate_number is None else plate_number
                    db_province = "No province" if province is None else province
                    
                    # Pass the properly formatted values to the database
                    db.update_incident(image_path, incident_id, db_plate_number, db_province)
                    logging.info(f"Updated incident {incident_id} with image {image_path}, plate {db_plate_number}, province {db_province}")
                except Exception as e:
                    logging.error(f"Failed to update incident: {e}", exc_info=True)
            # Upload image to Minio if output_path exists
            if output_path and os.path.exists(output_path):
                try:
                    # Use global Minio config variables 
                    global minio_endpoint, minio_access_key, minio_secret_key
                    if minio_endpoint and minio_access_key and minio_secret_key:
                        minio_client = MinioUpload(
                            endpoint=minio_endpoint,
                            access_key=minio_access_key,
                            secret_key=minio_secret_key,
                            secure=False  # Configure this as needed
                        )
                        
                        # Upload to Minio with appropriate naming
                        bucket_name = 'incident-image'  # You might want to make this configurable too
                        
                        # Use the same image path as in the database
                        if incident_id:
                            destination = str(incident_id) + ".jpg"
                            logging.info(f"Using image path {destination} for Minio upload")
                        else:
                            # Fallback to previous format if no incident_id is available
                            destination = f"plates_{object_id}.jpg"
                            logging.info(f"No incident ID available, using default naming format")
                        
                        if minio_client.upload_file(output_path, destination, bucket_name):
                            logging.info(f"Successfully uploaded image to Minio: {destination}")
                            result["minio_path"] = destination
                            
                            # Remove local file after successful upload
                            try:
                                os.remove(output_path)
                                logging.info(f"Removed local file: {output_path}")
                            except Exception as e:
                                logging.warning(f"Failed to remove local file {output_path}: {e}")
                        else:
                            logging.error(f"Failed to upload image to Minio: {output_path}")
                except Exception as e:
                    logging.error(f"Minio upload error: {e}", exc_info=True)
            
            # Calculate end-to-end latency
            if event_time:
                end_to_end_latency = time.time() - event_time
                result["end_to_end_latency"] = end_to_end_latency
            
            # Save results
            with file_lock:
                processed_results.append(result)
            
            # Add to detection logs with OCR output
            with detection_log_lock:
                ocr_log = {
                    "object_id": object_id,
                    "timestamp": time.time(),
                    "plate_number": plate_number if plate_number else None,
                    "province": province if province else None,
                    "ocr_success": bool(plate_number and plate_number != "Unknown"),
                    "ocr_time": ocr_time,
                    "total_processing_time": total_time,
                    "event_phase": "ocr_complete",
                    "minio_path": result.get("minio_path"),
                    "incident_id": incident_id,
                    "nvr_link": nvr_link
                }
                detection_logs.append(ocr_log)
            
            # Record metrics for monitoring
            update_metrics(processing_times, total_time)
            
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
                    
                    # Check if we have a marker_id and need to update RTSP URL
                    marker_id = event.get('marker_id')
                    if marker_id:
                        # Ensure we're using the right camera stream
                        update_rtsp_reader_for_marker(marker_id)
                    
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
                        # Gather nvr_link information if available
                        nvr_link = None
                        if marker_id:
                            nvr_link = get_rtsp_url_for_marker(marker_id)
                            
                        # New: Add detection to batch with nvr_link
                        detection_data = {
                            'plate_img': plate_img,
                            'object_id': object_id,
                            'event_time': event.get('first_seen', timestamp),
                            'timestamp': timestamp, 
                            'confidence': event.get('confidence', 0),
                            'detection_time': detection_time,
                            'processing_start': start_time,
                            'original_event': event.copy(),
                            'nvr_link': nvr_link
                        }
                        
                        detection_batch.append(detection_data)
                        logging.info(f"License plate detected for obj {object_id} in {detection_time*1000:.1f}ms, queued for OCR")
                        
                        if plate_img is not None:
                            logging.info(f"License plate detected for obj {object_id} in {detection_time*1000:.1f}ms, queued for OCR")
                        else:
                            logging.warning(f"No license plate detected for obj {object_id}, but updating incident anyway")
                        
                        # Process batch when it reaches size or when high priority
                        if len(detection_batch) >= batch_size or event.get('confidence', 0) > 0.8:
                            self._process_detection_batch(detection_batch, executor)
                            detection_batch = []
                            last_batch_time = time.time()
                            
                    else:
                        logging.warning(f"No license plate detected for obj {object_id}")
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
            
        # Submit only valid plate images to OCR, handle None cases directly
        future_to_detection = {}
        for detection in batch:
            if detection['plate_img'] is not None:
                # Only submit valid images to OCR
                future = executor.submit(
                    self.ocr_plate.predict, 
                    detection['plate_img'], 
                    detection['object_id']
                )
                future_to_detection[future] = detection
            else:
                # For None plate_img, directly call callback with "No number" and "No province"
                object_id = detection.get('object_id', 'unknown')
                # Skip OCR and directly handle as no plate detected
                self._ocr_result_callback(None, None, object_id, detection)
                logging.info(f"Directly handling None plate_img for obj {object_id} as undetected plate")
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_detection):
            detection = future_to_detection[future]
            try:
                plate_number, province = future.result()
                object_id = detection.get('object_id', 'unknown')
                
                # Don't modify None values - pass them through
                # (remove the sanitization code here)
                
                # Create a result record (even with unsuccessful OCR)
                event_data = detection
                self._ocr_result_callback(plate_number, province, object_id, event_data)
                
            except Exception as e:
                logging.error(f"Error in OCR task: {e}", exc_info=True)
                # For exceptions, still use None values
                object_id = detection.get('object_id', 'unknown')
                self._ocr_result_callback(None, None, object_id, detection)

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
    
    # Create monitor instance using global config path
    performance_monitor = PerformanceMonitor(config_path=performance_monitor_config_path)
    
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
            # Use global output_dir variable instead of config
            global output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "detection_log.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(detection_logs, f, ensure_ascii=False)
            logging.info(f"Saved {len(detection_logs)} detection logs to {output_file}")
            detection_log_last_save_time = time.time()
        except Exception as e:
            logging.error(f"Error saving detection logs: {e}")

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
    global detection_processor, processing_pipeline, frame_buffer, rtsp_reader
    
    try:
        # Load all configuration variables
        set_variable_from_config()
        
        # Initialize the frame buffer with buffer_size from config
        frame_buffer = TimeSyncedFrameBuffer(max_size=buffer_size)
        
        # Set main process to high priority
        set_high_priority()
        
        logging.info("Starting Enhanced License Plate Recognition System...")
        
        # Initialize plate processing pipeline using global config variables
        logging.info("Initializing plate processing pipeline")
        
        # Use the global variables for model paths
        global plate_weight, ocr_weight, default_rtsp_url
        
        # Determine optimal number of worker threads
        worker_threads = max(16, min(2, psutil.cpu_count(logical=False) - 1))
        logging.info(f"Using {worker_threads} worker threads for processing")
        
        processing_pipeline = PlateProcessingPipeline(
            plate_weight,
            ocr_weight,
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
        # Use MQTT config variables for DetectionLogProcessor
        logging.info("Initializing detection processor with real-time frame synchronization")
        global broker, port, topic, client_id, username, password, ca_cert, output_dir, default_rtsp_url
        
        # Create the detection processor directly using global variables instead of via a config file
        detection_processor = DetectionLogProcessor(
            mqtt_broker=broker,
            mqtt_port=port,
            detection_topic=topic,
            mqtt_username=username,
            mqtt_password=password,
            ca_cert_path=ca_cert,
            output_dir=output_dir,
            rtsp_url=default_rtsp_url,  # Start with default URL, will be updated dynamically
            snapshot_callback=mqtt_callback
        )
        
        # Store reference to rtsp_reader for updates
        rtsp_reader = detection_processor.rtsp_reader
        
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