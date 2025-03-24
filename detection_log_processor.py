import json
import os
import logging
import time
import cv2
import uuid
import numpy as np
import paho.mqtt.client as mqtt
from paho.mqtt import client as mqtt_client
import threading
from datetime import datetime
import queue
from rtsp_frame_reader import RTSPFrameReader
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DetectionLogProcessor:
    """
    High-performance detection log processor with precise frame synchronization
    """
    def __init__(self, config_path: str, snapshot_callback=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.output_dir = self.config.get('output_dir', 'event_images')
        os.makedirs(self.output_dir, exist_ok=True)
        self.snapshot_callback = snapshot_callback

        # Event queue for high-priority processing
        self.event_queue = queue.Queue()
        
        # Initialize RTSP reader
        rtsp_url = self.config.get('rtsp_url')
        if not rtsp_url:
            logging.error("RTSP URL is not provided in configuration.")
            raise ValueError("RTSP URL missing")
        
        self.rtsp_reader = RTSPFrameReader(rtsp_url)
        self.rtsp_reader.start()
        
        # Start event processor thread
        self.event_processor_thread = threading.Thread(
            target=self._event_processor, 
            daemon=True,
            name="Event-Processor"
        )
        self.event_processor_thread.start()

        # Setup MQTT client
        self.client_id = f"client-{uuid.uuid4()}"
        self.client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        self.client.username_pw_set(self.config.get('mqtt_username'), self.config.get('mqtt_password'))
        self.client.tls_set(ca_certs=self.config.get('ca_cert_path'))
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Connect with optimized settings
        self.client.connect(
            self.config.get('mqtt_broker'), 
            self.config.get('mqtt_port', 8883), 
            keepalive=15  # Reduced keepalive for faster error detection
        )
        
        self.mqtt_thread = threading.Thread(
            target=self._mqtt_loop, 
            daemon=True,
            name="MQTT-Client"
        )
        self.mqtt_thread.start()
        
        self.unique_id = self.config.get('unique_id', 'default_id')
        self.incident_topic = f"{self.config.get('detection_topic')}/{self.unique_id}"
        logging.info("Detection Log Processor initialized")

    def _mqtt_loop(self):
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Connected to MQTT broker, subscribing to {self.incident_topic}")
            self.client.subscribe(self.incident_topic)
        else:
            logging.error(f"Failed to connect to MQTT broker with code {rc}")

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages with high priority"""
        try:
            # Trigger burst mode for high framerate capture
            self.rtsp_reader.trigger_burst_mode(0.25)  # Capture at max framerate for 250ms
            
            # Parse payload
            payload = json.loads(msg.payload.decode())
            
            # Queue event for processing
            if msg.topic.startswith(self.incident_topic):
                # Add receipt timestamp
                payload['_receipt_time'] = time.time()
                self.event_queue.put(payload)
                logging.info(f"Queued detection event: {payload.get('event')} for object {payload.get('object_id')}")
                
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON payload: {msg.payload}")
        except Exception as e:
            logging.error(f"Error in MQTT message handler: {e}", exc_info=True)

    def _event_processor(self):
        """Process events from the queue with frame synchronization"""
        while True:
            try:
                # Get next event
                event = self.event_queue.get(timeout=1.0)
                
                # Process with precise timing
                self._process_detection_event_precise(event)
                
                # Mark as done
                self.event_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)  # Short sleep when no events
            except Exception as e:
                logging.error(f"Error in event processor: {e}", exc_info=True)
                time.sleep(0.1)  # Longer sleep on error

    def _process_detection_event_precise(self, event):
        """
        Process detection event with precise frame selection
        Uses frames captured during burst mode for maximum precision
        """
        start_time = time.time()
        
        # Extract timing information
        event_time = event.get('first_seen')
        receipt_time = event.get('_receipt_time')
        
        if isinstance(event_time, str):
            try:
                event_time = float(event_time)
            except ValueError:
                event_time = receipt_time
        elif event_time is None:
            event_time = receipt_time
        
        # Get all frames captured during burst mode
        burst_frames = self.rtsp_reader.get_event_frames()
        
        # Find the best frame - closest to the event time but not before it
        best_frame = None
        best_timestamp = 0
        best_time_diff = float('inf')
        best_quality = 0
        
        # First pass - find frames close to event time
        for frame, timestamp, quality in burst_frames:
            # Calculate time difference (prefer slightly later frames to earlier ones)
            # This bias compensates for network delay
            if timestamp < event_time:
                time_diff = (event_time - timestamp) * 1.5  # Higher penalty for too early frames
            else:
                time_diff = timestamp - event_time
                
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_frame = frame
                best_timestamp = timestamp
                best_quality = quality
        
        # If no good frame found in burst frames, find best frame in regular buffer
        if best_frame is None or best_time_diff > 0.5:  # If more than 500ms off
            logging.info("No suitable frame found in burst buffer, using regular frames")
            best_frame, best_timestamp = self.rtsp_reader.get_frame_with_timestamp()
            
        # If still no frame, give up
        if best_frame is None:
            logging.error("No valid frame found for event")
            return
        
        # Now process the frame with the bounding box
        event_type = event.get('event', 'unknown')
        object_id = event.get('object_id', 'unknown')
        
        # Format timestamp for filename
        timestamp_str = datetime.fromtimestamp(best_timestamp).strftime('%Y%m%d_%H%M%S')
        filename = f"{event_type}_obj{object_id}_{timestamp_str}.jpg"
        output_path = os.path.join(self.output_dir, filename)

        bbox = event.get('bbox')
        cropped = None
        
        if bbox and len(bbox) == 4:
            try:
                # Calculate bounding box coordinates
                x_center_norm, y_center_norm, width_norm, height_norm = bbox
                frame_height, frame_width = best_frame.shape[:2]
                x_center = x_center_norm * frame_width
                y_center = y_center_norm * frame_height
                width = width_norm * frame_width
                height = height_norm * frame_height
                
                # Calculate pixel coordinates with additional margin
                margin = 0.1  # 10% margin
                x1 = max(0, int(x_center - width/2 * (1+margin)))
                y1 = max(0, int(y_center - height/2 * (1+margin)))
                x2 = min(frame_width - 1, int(x_center + width/2 * (1+margin)))
                y2 = min(frame_height - 1, int(y_center + height/2 * (1+margin)))
                
                # Check that bounding box dimensions are valid
                if x2 <= x1 or y2 <= y1:
                    logging.error(f"Invalid bounding box: x1={x1}, x2={x2}, y1={y1}, y2={y2}. Using full frame.")
                    cropped = best_frame.copy()
                else:
                    # Create a copy of the display frame and the cropped region
                    display_frame = best_frame.copy()
                    cropped = best_frame[y1:y2, x1:x2].copy()
                    
                    # Draw bounding box on display frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{event_type.upper()}: Object {object_id}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Add timing information to image
                    frame_age = time.time() - best_timestamp
                    event_to_frame_diff = abs(best_timestamp - event_time) * 1000  # ms
                    cv2.putText(display_frame, 
                               f"Frame age: {frame_age*1000:.0f}ms | Event sync: {event_to_frame_diff:.0f}ms", 
                               (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save the display frame
                    if cv2.imwrite(output_path, display_frame):
                        logging.info(f"Saved event frame to {output_path}")
                    else:
                        logging.error("Failed to write event frame image")
            except Exception as e:
                logging.error(f"Error processing bounding box: {e}", exc_info=True)
                cropped = best_frame.copy()
        else:
            # No bounding box, use full frame
            cropped = best_frame.copy()

        # Prepare the cropped image for snapshot callback
        if self.snapshot_callback is not None and cropped is not None:
            try:
                # Convert to PIL Image
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cropped_rgb)
                
                # Add frame timestamp to event for traceability
                event['frame_timestamp'] = best_timestamp
                event['processing_latency'] = time.time() - receipt_time
                
                # Call snapshot callback
                self.snapshot_callback(pil_image, event)
            except Exception as e:
                logging.error(f"Error preparing cropped image for callback: {e}", exc_info=True)
                
        # Log performance metrics
        process_time = time.time() - start_time
        logging.info(f"Event processing completed in {process_time*1000:.1f}ms, "
                    f"frame sync precision: {best_time_diff*1000:.1f}ms")