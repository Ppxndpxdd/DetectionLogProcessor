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
from rtsp_frame_reader import RTSPFrameReader
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DetectionLogProcessor:
    """
    Processes detection logs received via MQTT.
    When a detection event is received, it retrieves the latest RTSP frame,
    draws the bounding box, saves the frame, and then calls a snapshot callback
    with a cropped region for further processing.
    """
    def __init__(self, config_path: str, snapshot_callback=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.output_dir = self.config.get('output_dir', 'event_images')
        os.makedirs(self.output_dir, exist_ok=True)
        self.snapshot_callback = snapshot_callback

        self.client_id = f"client-{uuid.uuid4()}"
        self.client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        self.client.username_pw_set(self.config.get('mqtt_username'), self.config.get('mqtt_password'))
        self.client.tls_set(ca_certs=self.config.get('ca_cert_path'))
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.config.get('mqtt_broker'), self.config.get('mqtt_port', 8883), 60)
        self.mqtt_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
        self.mqtt_thread.start()
        self.unique_id = self.config.get('unique_id', 'default_id')
        self.incident_topic = f"{self.config.get('detection_topic')}/{self.unique_id}"
        logging.info("Detection Log Processor initialized")

        rtsp_url = self.config.get('rtsp_url')
        if not rtsp_url:
            logging.error("RTSP URL is not provided in configuration.")
            raise ValueError("RTSP URL missing")
        # You can set use_gstreamer=True for lower latency and to avoid buffer warnings.
        self.rtsp_reader = RTSPFrameReader(rtsp_url)
        self.rtsp_reader.start()

    def _mqtt_loop(self):
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            self.client.subscribe(self.incident_topic)
        else:
            logging.error(f"Failed to connect to MQTT broker with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == self.config.get('rtsp_reference_topic', 'stream_log'):
                logging.info("Received RTSP stream reference (used for logging only).")
            elif msg.topic.startswith(self.incident_topic):
                self.handle_detection_event(payload)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON payload: {msg.payload}")
        except Exception as e:
            logging.error(f"Error processing MQTT message: {e}")

    def handle_detection_event(self, payload):
        logging.info(f"Received detection event via MQTT: {json.dumps(payload, indent=2)}")
        logging.info(f"Detection event: {payload.get('event')} for object {payload.get('object_id')}")
        self.process_detection_event(payload)

    def process_detection_event(self, event):
        frame = self.rtsp_reader.get_frame()  # Now returns just the frame
        if frame is None or not isinstance(frame, np.ndarray):
            logging.error("No valid frame available from RTSP stream at this moment.")
            return

        # Verify frame is a valid numpy array with shape
        if not hasattr(frame, 'shape') or len(frame.shape) < 2:
            logging.error(f"Invalid frame format: {type(frame)}")
            return
            
        event_type = event.get('event', 'unknown')
        object_id = event.get('object_id', 'unknown')
        event_time = event.get('first_seen') or time.time()
        if isinstance(event_time, str):
            try:
                event_time = float(event_time)
            except ValueError:
                event_time = time.time()
        timestamp_str = datetime.fromtimestamp(event_time).strftime('%Y%m%d_%H%M%S')
        filename = f"{event_type}_obj{object_id}_{timestamp_str}.jpg"
        output_path = os.path.join(self.output_dir, filename)

        bbox = event.get('bbox')
        cropped = None
        if bbox and len(bbox) == 4:
            try:
                x_center_norm, y_center_norm, width_norm, height_norm = bbox
                frame_height, frame_width = frame.shape[:2]
                x_center = x_center_norm * frame_width
                y_center = y_center_norm * frame_height
                width = width_norm * frame_width
                height = height_norm * frame_height
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width - 1, x2)
                y2 = min(frame_height - 1, y2)
                
                # Check that bounding box dimensions are valid
                if x2 <= x1 or y2 <= y1:
                    logging.error(f"Invalid bounding box: x1={x1}, x2={x2}, y1={y1}, y2={y2}. Using full frame.")
                    cropped = frame.copy()  # Use a copy of the full frame
                else:
                    cropped = frame[y1:y2, x1:x2].copy()  # Make a copy of the slice
                    
                # Add the bounding box to the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{event_type.upper()}: Object {object_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                logging.error(f"Error drawing bounding box: {e}", exc_info=True)
                # If we failed to process the bounding box, use a copy of the full frame
                cropped = frame.copy()

        # Save the frame (with bounding box if available)
        try:
            if cv2.imwrite(output_path, frame):
                logging.info(f"Saved event frame to {output_path}")
            else:
                logging.error("Failed to write event frame image.")
        except Exception as e:
            logging.error(f"Exception saving event frame: {e}", exc_info=True)

        # Prepare the cropped image for snapshot callback
        if self.snapshot_callback is not None:
            if cropped is None:
                cropped = frame.copy()  # Use a copy of the full frame
                
            # Ensure cropped is a valid numpy array
            if not isinstance(cropped, np.ndarray):
                logging.error(f"Invalid cropped image type: {type(cropped)}")
                return
                
            if not hasattr(cropped, "shape") or cropped.size == 0:
                logging.error("Cropped image is empty or has an invalid shape.")
                return

            try:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cropped_rgb)
                self.snapshot_callback(pil_image, event)
            except Exception as e:
                logging.error(f"Error preparing cropped image for callback: {e}", exc_info=True)

    def run(self):
        try:
            print(self.incident_topic)
            logging.info("Detection Log Processor running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Detection Log Processor stopping")
        finally:
            self.client.disconnect()
            self.rtsp_reader.stop()
