import logging
import time
import cv2
import threading
import os
import queue
import numpy as np

class RTSPFrameReader(threading.Thread):
    """
    Enhanced RTSP frame reader with aggressive frame skipping and timestamp tracking
    to ensure processing only the most recent frames.
    """
    def __init__(self, rtsp_url: str, use_gstreamer: bool = False, max_queue_size: int = 1):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.use_gstreamer = use_gstreamer
        self.max_queue_size = max_queue_size
        
        # Frame buffer with timestamps
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.last_frame_time = 0
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        
        # Connection parameters
        self.reconnect_interval = 3  # seconds
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0
        
        self._setup_capture()
        self.latest_frame = None
        self.latest_frame_time = 0
        self.lock = threading.Lock()
        self.stopped = False

    def _setup_capture(self):
        if self.use_gstreamer:
            # Low-latency GStreamer pipeline
            pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=0 buffer-mode=auto ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # FFMPEG with optimized parameters
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024|max_delay;0|reorder_queue_size;0"
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            # self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Minimal buffering settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
        if not self.cap.isOpened():
            logging.error(f"RTSPFrameReader: Failed to open RTSP stream: {self.rtsp_url}")

    def run(self):
        last_log_time = time.time()
        frames_since_log = 0
        
        while not self.stopped:
            try:
                # Skip all buffered frames - only get the newest one
                for _ in range(5):  # Try to skip several frames in the queue
                    self.cap.grab()
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    current_time = time.time()
                    self.frame_count += 1
                    frames_since_log += 1
                    
                    # Update the latest frame with timestamp
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_frame_time = current_time
                    
                    # Log performance metrics every 5 seconds
                    if current_time - last_log_time > 5:
                        fps = frames_since_log / (current_time - last_log_time)
                        logging.debug(f"RTSP Reader: {fps:.2f} FPS, Frame count: {self.frame_count}, "
                                    f"Processed: {self.processed_count}, Skipped: {self.skipped_count}")
                        last_log_time = current_time
                        frames_since_log = 0
                        
                    # Small sleep to prevent CPU overload
                    time.sleep(0.001)
                else:
                    logging.warning("RTSPFrameReader: No frame received, reconnecting...")
                    time.sleep(0.5)
                    self._reconnect()
            except Exception as e:
                logging.error(f"RTSPFrameReader: Exception in run loop: {e}")
                time.sleep(0.5)
                self._reconnect()

    def _reconnect(self):
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logging.error(f"RTSPFrameReader: Max reconnection attempts reached for {self.rtsp_url}")
            time.sleep(self.reconnect_interval * 2)
            self.reconnect_attempts = 0
            
        try:
            self.cap.release()
        except Exception as e:
            logging.error(f"RTSPFrameReader: Error releasing capture: {e}")
            
        time.sleep(self.reconnect_interval)
        logging.info(f"RTSPFrameReader: Reconnecting to {self.rtsp_url}, attempt {self.reconnect_attempts}")
        self._setup_capture()
        
        if self.cap.isOpened():
            self.reconnect_attempts = 0
            logging.info(f"RTSPFrameReader: Successfully reconnected to {self.rtsp_url}")

    def get_frame(self):
        """Get the most recent frame"""
        with self.lock:
            if self.latest_frame is not None:
                self.processed_count += 1
                # Return just the frame, not a tuple
                return self.latest_frame.copy()
            return None
            
    def get_frame_with_timestamp(self):
        """Get the most recent frame with timestamp"""
        with self.lock:
            if self.latest_frame is not None:
                self.processed_count += 1
                # Return both frame and timestamp as a tuple
                return self.latest_frame.copy(), self.latest_frame_time
            return None, 0

    def stop(self):
        self.stopped = True
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()