import json
import os
import logging
import time
import cv2
import uuid
import paho.mqtt.client as mqtt
from paho.mqtt import client as mqtt_client
import threading
from datetime import datetime
from rtsp_frame_reader import RTSPFrameReader
from PIL import Image  # Added to support snapshot conversion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DetectionLogProcessor:
    """
    Processes detection logs received via MQTT.
    When an event is received (for events like no_entry, no_parking, wrong_way),
    it retrieves the latest frame from a persistent RTSP stream connection.
    After processing the event (drawing bounding box and saving the frame),
    it optionally sends a snapshot (as a PIL image) via a callback.
    """
    def __init__(self, config_path: str, snapshot_callback=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # For properties, we still use the local video file (if needed)
        self.output_dir = self.config.get('output_dir', 'event_images')
        os.makedirs(self.output_dir, exist_ok=True)
        self.snapshot_callback = snapshot_callback

        # Initialize MQTT client
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

        # Start a persistent RTSP connection using the provided RTSP URL.
        rtsp_url = self.config.get('rtsp_url')
        if not rtsp_url:
            logging.error("RTSP URL is not provided in configuration.")
            raise ValueError("RTSP URL missing")
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
        frame = self.rtsp_reader.get_frame()
        if frame is None:
            logging.error("No frame available from RTSP stream at this moment.")
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{event_type.upper()}: Object {object_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cropped = frame[y1:y2, x1:x2]
            except Exception as e:
                logging.error(f"Error drawing bounding box: {e}")

        if cv2.imwrite(output_path, frame):
            logging.info(f"Saved event frame to {output_path}")
        else:
            logging.error("Failed to write event frame image.")

        # Call snapshot callback with both the image and event
        if self.snapshot_callback is not None:
            if cropped is None:
                cropped = frame
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)
            self.snapshot_callback(pil_image, event)

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

if __name__ == "__main__":
    config_path = "config.json"
    processor = DetectionLogProcessor(config_path)
    processor.run()
