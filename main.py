import time
import threading
import os
import json
from PIL import Image
import concurrent.futures

from detection_log_processor import DetectionLogProcessor  
from plate_detector import PlateDetector
from ocr_plate import OCRPlate

# --- Load configuration from config.json ---
with open("config.json", "r") as f:
    config = json.load(f)

# Create output directory if it doesn't exist
os.makedirs(config["output_dir"], exist_ok=True)

# Global lists for logging
detection_log_list = []
merged_output_list = []

# Lock for thread-safe file writes
file_lock = threading.Lock()

# --- Initialize PlateDetector and OCRPlate using model paths from config ---
plate_detector_model_path = config['plate_detector_model_path']  # Set in config.json
ocr_plate_model_path = config['ocr_plate_model_path']            # Set in config.json
plate_detector = PlateDetector(plate_detector_model_path)
ocr_plate = OCRPlate(ocr_plate_model_path)

# --- Create a ThreadPoolExecutor for heavy processing tasks ---
# The heavy processing (plate detection and OCR) will be offloaded here.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def process_snapshot(event, snapshot):
    """
    Worker function to perform heavy processing.
    It runs plate detection and OCR on the snapshot, then merges the results
    into the detection event. Finally, it writes both the raw detection log
    and merged output to JSON files.
    """
    # Perform plate detection on the snapshot image.
    plate_region = plate_detector.detect_plate(snapshot)
    if plate_region:
        # Run OCR on the detected plate region.
        plate_text, province = ocr_plate.predict(plate_region)
        print(f"OCR Result: Plate: {plate_text}, Province: {province}")
        event["plate_text"] = plate_text
        event["province"] = province
    else:
        print("Plate not detected.")
        event["plate_text"] = None
        event["province"] = None

    # Append the merged event to the output list.
    merged_output_list.append(event)
    # Write both the raw detection log and merged output to JSON files.
    with file_lock:
        with open("detection_log.json", "w") as f:
            json.dump(detection_log_list, f, indent=2)
        with open("output.json", "w") as f:
            json.dump(merged_output_list, f, indent=2)

def snapshot_callback(snapshot, event):
    """
    This callback is called by DetectionLogProcessor as soon as a detection event occurs.
    It immediately captures the current snapshot (already a PIL image) and dispatches
    the heavy processing to the ThreadPoolExecutor.
    """
    # Immediately record the detection event.
    detection_log_list.append(event)
    # Dispatch heavy processing (plate detection/OCR) to the executor.
    executor.submit(process_snapshot, event, snapshot)
    print("Snapshot dispatched to worker pool for processing.")

# --- Initialize DetectionLogProcessor with the snapshot callback ---
# This instance sets up the MQTT client and RTSPFrameReader.
detection_processor = DetectionLogProcessor("config.json", snapshot_callback)

# The DetectionLogProcessor internally runs its own MQTT loop and RTSPFrameReader in separate threads.
# Because the snapshot_callback immediately offloads heavy processing, the RTSPFrameReader remains responsive.

# --- Keep the main thread alive ---
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    detection_processor.client.disconnect()
    detection_processor.rtsp_reader.stop()
    executor.shutdown(wait=True)
