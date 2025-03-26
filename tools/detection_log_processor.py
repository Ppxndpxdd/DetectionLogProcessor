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
from tools.rtsp_frame_reader import RTSPFrameReader
from PIL import Image
# Import database connection module
import database_connect.postgres as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DetectionLogProcessor:
    """
    High-performance detection log processor with precise frame synchronization
    """
    def __init__(self, mqtt_broker, mqtt_port, detection_topic, mqtt_username, mqtt_password, 
                 ca_cert_path, output_dir, rtsp_url, snapshot_callback=None):
        """
        Initialize with direct configuration parameters instead of reading from file
        """
        # Store configuration parameters directly
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.detection_topic = detection_topic
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.ca_cert_path = ca_cert_path
        self.output_dir = output_dir
        self.rtsp_url = rtsp_url
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.snapshot_callback = snapshot_callback

        # Event queue for high-priority processing
        self.event_queue = queue.Queue()
        
        # Initialize RTSP reader
        if not self.rtsp_url:
            logging.error("RTSP URL is not provided in configuration.")
            raise ValueError("RTSP URL missing")
        
        self.rtsp_reader = RTSPFrameReader(self.rtsp_url)
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
        self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
        self.client.tls_set(ca_certs=self.ca_cert_path)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Connect with optimized settings
        self.client.connect(
            self.mqtt_broker, 
            self.mqtt_port, 
            keepalive=15  # Reduced keepalive for faster error detection
        )
        
        self.mqtt_thread = threading.Thread(
            target=self._mqtt_loop, 
            daemon=True,
            name="MQTT-Client"
        )
        self.mqtt_thread.start()
            
        logging.info("Detection Log Processor initialized")

    def _mqtt_loop(self):
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Connected to MQTT broker, subscribing to {self.detection_topic}")
            self.client.subscribe(self.detection_topic)
        else:
            logging.error(f"Failed to connect to MQTT broker with code {rc}")

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages with enhanced frame synchronization"""
        try:
            # Parse payload
            payload = json.loads(msg.payload.decode())
            
            # Add receipt timestamp with high precision
            receipt_time = time.time()
            payload['_received_time'] = receipt_time
            
            # Extract event time with fallback mechanisms
            event_time = payload.get('first_seen')
            if isinstance(event_time, str):
                try:
                    event_time = float(event_time)
                except ValueError:
                    event_time = receipt_time
            elif event_time is None:
                event_time = receipt_time
                
            # Log the initial detection event
            from main import detection_logs, detection_log_lock
            with detection_log_lock:
                initial_log = {
                    "object_id": payload.get("object_id", "unknown"),
                    "timestamp": receipt_time,
                    "event_time": event_time,
                    "event_type": payload.get("event", "unknown"),
                    "detection_confidence": payload.get("confidence", 0),
                    "mqtt_receipt_time": receipt_time,
                    "event_phase": "initial_detection",
                    "raw_payload": payload.copy()
                }
                detection_logs.append(initial_log)
                
            # Calculate estimated network delay for better synchronization
            network_delay = 0.05  # Assume 50ms network delay
            adjusted_event_time = event_time - network_delay
            
            # Trigger precise burst with extended pre-event time
            self.rtsp_reader.trigger_precise_burst(
                event_time=adjusted_event_time,
                duration=0.3,       # Capture slightly longer after event
                pre_event_time=0.2  # Capture more time before event
            )
            
            # Calculate event priority for processing order
            priority = self._calculate_event_priority(payload)
            
            # Apply backpressure if system is overloaded
            if self.event_queue.qsize() > 15:
                # Log detailed backlog information
                logging.warning(f"Event queue backlog: {self.event_queue.qsize()} events, applying backpressure")
                
                # Skip low priority events when system is very busy
                if self.event_queue.qsize() > 25 and priority < 7:
                    logging.warning(f"Dropping low-priority event due to severe backlog")
                    return
                    
            # Add processing metadata
            payload['_priority'] = priority
            payload['_adjusted_event_time'] = adjusted_event_time
            
            # Queue with priority and timestamps
            self.event_queue.put((priority, receipt_time, payload))
            
            if priority >= 8:  # High priority event
                logging.info(f"Queued HIGH PRIORITY event: {payload.get('event')} "
                           f"for object {payload.get('object_id')}, priority {priority}")
            else:
                logging.info(f"Queued event: {payload.get('event')} "
                           f"for object {payload.get('object_id')}, priority {priority}")
                
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON payload: {msg.payload}")
        except Exception as e:
            logging.error(f"Error in MQTT message handler: {e}", exc_info=True)
            
    def _calculate_event_priority(self, event):
        """Calculate priority of event based on content (1-10, 10 highest)"""
        # Base priority
        priority = 5
        
        # Increase priority for certain event types
        event_type = event.get('event', '').lower()
        if 'license' in event_type or 'plate' in event_type:
            priority += 3
        elif 'vehicle' in event_type:
            priority += 2
        
        # Higher priority for events with high confidence
        confidence = event.get('confidence', 0)
        if confidence > 0.8:
            priority += 1
        
        # Ensure within range
        return max(1, min(10, priority))

    def _event_processor(self):
        """Process events from the queue with frame synchronization"""
        while True:
            try:
                # Get next event (with timeout to prevent blocking)
                priority, receipt_time, event = self.event_queue.get(timeout=1.0)
                
                # Check if event is too old (5 seconds max age)
                current_time = time.time()
                if current_time - receipt_time > 5.0:
                    logging.warning(f"Discarding stale event received {current_time - receipt_time:.1f}s ago")
                    self.event_queue.task_done()
                    continue
                    
                # Report queue backlog for performance monitoring
                queue_size = self.event_queue.qsize()
                if queue_size > 0:
                    logging.info(f"Event queue backlog: {queue_size} events")
                
                # Process with precise timing and get output path
                output_path = self._process_detection_event_precise(event)
                
                # Store output path in detection logs if available
                if output_path:
                    from main import detection_logs, detection_log_lock
                    with detection_log_lock:
                        # Find the matching log entry and update it
                        object_id = event.get('object_id', 'unknown')
                        for log in detection_logs:
                            if log.get('object_id') == object_id and log.get('event_phase') == 'initial_detection':
                                log['output_image_path'] = output_path
                                break
                
                # Mark as done
                self.event_queue.task_done()
                
            except queue.Empty:
                # Short sleep when no events
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error in event processor: {e}", exc_info=True)
                time.sleep(0.1)  # Longer sleep on error

    def _process_detection_event_precise(self, event):
        """
        Process detection event with precise frame selection
        Uses frames captured during burst mode for maximum precision
        
        Returns:
            str: Path to the saved output image file
        """
        start_time = time.time()
        output_path = None
        incident_id = None  # To store the database incident ID
        
        try:
            # First, insert the incident into the database to get an incident_id
            # Following the format from incident_subscribe.py
            try:
                incident_id = db.insert_incident_log(
                    event.get('object_id', 'unknown'),
                    event.get('class_id', 0),
                    event.get('confidence', 0),
                    event.get('marker_id', 0),
                    event.get('first_seen', time.time()),
                    event.get('last_seen', time.time()),
                    event.get('event', 'unknown'),
                    event.get('bbox', []),
                    event.get('id_rule_applied', 0)
                )
                logging.info(f"Created incident ID {incident_id} for object {event.get('object_id', 'unknown')}")
                
                # Add incident_id to event data for later use
                event['incident_id'] = incident_id
                
                # Get NVR link associated with the marker_id
                try:
                    marker_id = event.get('marker_id', 0)
                    nvr_link = db.get_nvr_link_by_marker_id(marker_id)
                    if nvr_link:
                        logging.info(f"Found NVR link for marker_id {marker_id}: {nvr_link}")
                        # Add NVR link to event data for later use (e.g., in OCR processing)
                        event['nvr_link'] = nvr_link
                        
                        # Update the rtsp_reader to use this NVR link
                        if self.rtsp_reader and nvr_link != self.rtsp_reader.rtsp_url:
                            try:
                                logging.info(f"Switching RTSP stream to {nvr_link} for marker {marker_id}")
                                self.rtsp_reader.update_rtsp_url(nvr_link)
                            except Exception as e:
                                logging.error(f"Failed to update RTSP URL: {e}", exc_info=True)
                    else:
                        logging.warning(f"No NVR link found for marker_id {marker_id}")
                except Exception as e:
                    logging.error(f"Error getting NVR link: {e}", exc_info=True)
                
            except Exception as e:
                logging.error(f"Failed to insert incident log: {e}", exc_info=True)
                # Continue processing even if database insertion fails
        
            # Extract timing information
            event_time = event.get('first_seen')
            # Fix: Use the correct key name and add fallback if timestamp is missing
            receipt_time = event.get('_received_time')
            
            if receipt_time is None:
                # Fallback if _received_time is not available
                receipt_time = event.get('_receipt_time')
                
            if receipt_time is None:
                # Ultimate fallback - use current time
                receipt_time = time.time()
                logging.warning(f"No receipt timestamp found for event {event.get('object_id', 'unknown')}")
            
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
                return None
            
            # Now process the frame with the bounding box
            event_type = event.get('event', 'unknown')
            object_id = event.get('object_id', 'unknown')
            
            # Format timestamp for filename
            timestamp_str = datetime.fromtimestamp(best_timestamp).strftime('%Y%m%d_%H%M%S')
            
            # Use incident_id in the filename if available
            if incident_id:
                filename = str(incident_id) + ".jpg"
            else:
                filename = f"plates_{object_id}.jpg"
                
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
                        
                        # Add incident ID to image if available
                        if incident_id:
                            cv2.putText(display_frame, 
                                      f"Incident ID: {incident_id}", 
                                      (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Save the display frame
                        if cv2.imwrite(output_path, display_frame):
                            logging.info(f"Saved event frame to {output_path}")
                        else:
                            logging.error("Failed to write event frame image")
                            output_path = None
                except Exception as e:
                    logging.error(f"Error processing bounding box: {e}", exc_info=True)
                    cropped = best_frame.copy()
                    output_path = None
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
                    # Safe calculation of processing latency
                    event['processing_latency'] = time.time() - receipt_time
                    
                    # Add output path and incident_id to event data for reference in main
                    event['output_image_path'] = output_path
                    if incident_id:
                        event['incident_id'] = incident_id
                    
                    # Call snapshot callback
                    self.snapshot_callback(pil_image, event)
                except Exception as e:
                    logging.error(f"Error preparing cropped image for callback: {e}", exc_info=True)
                    
            # Log performance metrics
            process_time = time.time() - start_time
            logging.info(f"Event processing completed in {process_time*1000:.1f}ms, "
                        f"frame sync precision: {best_time_diff*1000:.1f}ms")
                        
            return output_path
            
        except Exception as e:
            logging.error(f"Error in process_detection_event_precise: {e}", exc_info=True)
            return None