import logging
import time
import cv2
import threading
import os
import queue
import numpy as np
from collections import deque
import platform

class RTSPFrameReader(threading.Thread):
    """
    High-performance RTSP frame reader with burst capture for critical events
    """
    def __init__(self, rtsp_url, use_gstreamer=None, max_queue_size=1):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        
        # Auto-detect optimal backend based on platform
        if use_gstreamer is None:
            self.use_gstreamer = platform.system() == "Linux" and cv2.getBuildInformation().find("GStreamer") != -1
        else:
            self.use_gstreamer = use_gstreamer
            
        self.max_queue_size = max_queue_size
        
        # Enhanced frame buffer with adaptive timing
        self.frame_deque = deque(maxlen=30)  # Store more frames for better matching
        
        # Critical event handling - separate high-priority buffer for events
        self.event_frames = deque(maxlen=15)  # Special buffer for event frames
        self.burst_mode = False  # Flag for burst mode during events
        self.burst_until = 0  # Time when burst mode should end
        self.burst_frame_count = 0  # Number of frames captured in burst mode
        
        # Performance tracking
        self.last_frame_time = 0
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.fps = 0
        
        # Frame quality metrics
        self.frame_sharpness = 0
        self.frame_brightness = 0
        self.motion_score = 0
        
        # Adaptive parameters
        self.quality_threshold = 0.4  # Lower threshold for faster response
        self.skip_frames_count = 1    # Minimize frame skipping for events
        
        # Connection parameters
        self.reconnect_interval = 3
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0
        
        # Performance monitoring
        self.frame_interval_history = deque(maxlen=100)
        self.last_fps_calc_time = time.time()
        self.frames_since_fps_calc = 0
        
        self._setup_capture()
        self.latest_frame = None
        self.latest_frame_time = 0
        self.lock = threading.RLock()
        self.event_lock = threading.RLock()  # Separate lock for event frames
        self.stopped = False
        
        # Frame prediction
        self.prev_frame = None
        self.motion_vectors = None

    def _setup_capture(self):
        """Configure video capture with optimal settings for platform"""
        if self.use_gstreamer:
            # Ultra-low latency GStreamer pipeline
            pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=0 buffer-mode=none drop-on-latency=true ! "
                "rtph264depay ! h264parse ! avdec_h264 max-threads=4 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            logging.info("Using GStreamer backend for RTSP")
        else:
            # Configure FFMPEG with optimal parameters
            transport = "udp" if self.rtsp_url.startswith("rtsp://") else "tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|"
                "buffer_size;16384|"
                "max_delay;0|"
                "stimeout;2000000|"
                "reorder_queue_size;0"
            )
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Configure for minimal latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            
            # Try hardware acceleration if available
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    logging.info("CUDA hardware acceleration available")
            except:
                pass
            
            logging.info("Using FFMPEG backend for RTSP")
        
        if self.cap.isOpened():
            logging.info(f"RTSP stream opened: {self.rtsp_url}")
            
            # Adjust capture parameters for minimal latency
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logging.info(f"Stream properties: {width}x{height} @ {fps}fps")
            
            # Set timeouts for faster frame capture
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            logging.error(f"Failed to open RTSP stream: {self.rtsp_url}")

    def _calculate_frame_quality(self, frame):
        """Fast calculation of frame quality metrics"""
        if frame is None:
            return 0
            
        try:
            # Faster quality assessment - only process a center crop
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            crop_size = min(width, height) // 4
            
            # Extract center region for analysis
            center_crop = frame[
                max(0, center_y - crop_size):min(height, center_y + crop_size),
                max(0, center_x - crop_size):min(width, center_x + crop_size)
            ]
            
            # Calculate using grayscale to save time
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            
            # Fast sharpness calculation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            self.frame_sharpness = laplacian.var()
            
            # Fast brightness calculation
            self.frame_brightness = np.mean(gray)
            
            # Fast motion detection if we have previous frame
            if self.prev_frame is not None and self.prev_frame.shape == gray.shape:
                diff = cv2.absdiff(gray, self.prev_frame)
                self.motion_score = np.mean(diff)
            else:
                self.motion_score = 0
                
            self.prev_frame = gray.copy()
                
            # Combined quality score (0-1)
            if self.frame_sharpness > 100:  # Lower threshold for faster response
                quality = min(1.0, self.frame_sharpness / 1000.0) * 0.6 + self.motion_score * 0.4
            else:
                quality = 0.2 + self.motion_score * 0.8
                
            return quality
            
        except Exception as e:
            logging.error(f"Error calculating frame quality: {e}")
            return 0.5  # Default quality

    def trigger_burst_mode(self, duration=0.25):
        """
        Trigger burst mode to capture frames at maximum rate for a short duration
        This is called when an MQTT event arrives
        """
        with self.event_lock:
            self.burst_mode = True
            self.burst_until = time.time() + duration
            self.burst_frame_count = 0
            self.event_frames.clear()  # Clear previous event frames
        logging.debug(f"Burst mode triggered for {duration}s")

    def trigger_precise_burst(self, event_time, duration=0.5, pre_event_time=0.2):
        """
        Enhanced burst mode that captures frames around a specific event time
        
        Args:
            event_time: Timestamp of the event
            duration: How long to continue burst after event_time (seconds)
            pre_event_time: How much time before event to include (seconds)
        """
        with self.event_lock:
            self.burst_mode = True
            self.burst_start_time = event_time - pre_event_time
            self.burst_until = event_time + duration
            self.burst_frame_count = 0
            self.burst_event_time = event_time
            
            # Don't clear existing event frames if this event is close to previous one
            if len(self.event_frames) == 0 or abs(event_time - self.last_burst_event_time) > 1.0:
                self.event_frames.clear()
                
            self.last_burst_event_time = event_time
            
        logging.debug(f"Precise burst mode triggered for event at {event_time}, "
                     f"capturing frames from {self.burst_start_time} to {self.burst_until}")

    def run(self):
        """Main acquisition thread - continuously captures frames with adaptive timing"""
        last_frame_time = 0
        last_log_time = time.time()
        frames_since_log = 0
        consecutive_empty_frames = 0
        
        while not self.stopped:
            try:
                # Check if we should be in burst mode
                in_burst_mode = False
                with self.event_lock:
                    if self.burst_mode and time.time() <= self.burst_until:
                        in_burst_mode = True
                    elif self.burst_mode:
                        self.burst_mode = False
                        logging.debug(f"Burst mode ended, captured {self.burst_frame_count} frames")
                
                # In burst mode, capture every frame with no skipping
                if in_burst_mode:
                    skip_count = 0
                else:
                    # Regular mode - use adaptive skip count
                    skip_count = max(0, self.skip_frames_count - 1)
                
                # Skip frames if needed
                for _ in range(skip_count):
                    self.cap.grab()
                
                # Read the frame
                ret, frame = self.cap.read()
                current_time = time.time()
                
                if ret and frame is not None:
                    # Track FPS
                    self.frames_since_fps_calc += 1
                    frames_since_log += 1
                    self.frame_count += 1
                    
                    if last_frame_time > 0:
                        interval = current_time - last_frame_time
                        self.frame_interval_history.append(interval)
                    
                    # Calculate quality 
                    frame_quality = self._calculate_frame_quality(frame)
                    
                    # For burst mode, capture every frame regardless of quality
                    if in_burst_mode:
                        with self.event_lock:
                            self.event_frames.append((frame.copy(), current_time, frame_quality))
                            self.burst_frame_count += 1
                            
                            # Also update latest frame reference
                            self.latest_frame = frame.copy()
                            self.latest_frame_time = current_time
                    
                    # For regular mode, filter by quality
                    elif frame_quality >= self.quality_threshold or consecutive_empty_frames >= 5:
                        # Store frame with timestamp and quality info
                        with self.lock:
                            self.latest_frame = frame.copy()
                            self.latest_frame_time = current_time
                            self.frame_deque.append((frame.copy(), current_time, frame_quality))
                        consecutive_empty_frames = 0
                    else:
                        self.skipped_count += 1
                        consecutive_empty_frames += 1
                    
                    # Log performance metrics every 5 seconds
                    if current_time - last_log_time > 5:
                        elapsed = current_time - last_log_time
                        if elapsed > 0:
                            fps = frames_since_log / elapsed
                            self.fps = fps
                            logging.debug(f"RTSP: {fps:.1f}fps, Quality: {frame_quality:.2f}, "
                                          f"Buffer: {len(self.frame_deque)}, "
                                          f"Event Buffer: {len(self.event_frames)}")
                            frames_since_log = 0
                            last_log_time = current_time
                    
                    last_frame_time = current_time
                else:
                    consecutive_empty_frames += 1
                    
                    if consecutive_empty_frames > 20:
                        logging.warning(f"Multiple empty frames ({consecutive_empty_frames}), reconnecting...")
                        self._reconnect()
                        consecutive_empty_frames = 0
                
                # Sleep adaptively based on mode
                if in_burst_mode:
                    # Minimal sleep in burst mode
                    time.sleep(0.001)
                else:
                    # Normal mode - adaptive sleep
                    if len(self.frame_interval_history) > 0:
                        expected_interval = np.median(self.frame_interval_history) * 0.5
                        sleep_time = max(0.001, min(0.01, expected_interval))
                    else:
                        sleep_time = 0.001
                    time.sleep(sleep_time)
                
            except Exception as e:
                logging.error(f"Error in RTSP reader: {e}", exc_info=True)
                self._reconnect()
                time.sleep(0.5)

    def _reconnect(self):
        """Reconnect to the RTSP stream with backoff strategy"""
        self.reconnect_attempts += 1
        
        try:
            self.cap.release()
        except Exception as e:
            logging.error(f"Error releasing capture: {e}")
            
        # Exponential backoff for reconnection attempts
        backoff_time = min(self.reconnect_interval * (1.5 ** min(self.reconnect_attempts, 5)), 20)
        logging.info(f"Reconnecting to {self.rtsp_url} in {backoff_time:.1f}s (attempt {self.reconnect_attempts})")
        time.sleep(backoff_time)
        
        self._setup_capture()
        
        if self.cap.isOpened():
            self.reconnect_attempts = 0
            logging.info(f"Successfully reconnected to RTSP stream")
            
            # Clear frame history after reconnect
            with self.lock:
                self.frame_deque.clear()
                self.frame_interval_history.clear()
                
            with self.event_lock:
                self.event_frames.clear()
        else:
            logging.error(f"Failed to reconnect on attempt {self.reconnect_attempts}")

    def get_best_frame(self):
        """Get the best quality recent frame"""
        with self.lock:
            if not self.frame_deque:
                return None, 0
                
            # Find highest quality frame from recent frames
            best_frame, best_time, best_quality = max(self.frame_deque, key=lambda x: x[2])
            self.processed_count += 1
            
            return best_frame.copy(), best_time
            
    def get_event_frames(self):
        """Get all frames captured during burst mode"""
        with self.event_lock:
            if not self.event_frames:
                return []
            return [(frame.copy(), timestamp, quality) for frame, timestamp, quality in self.event_frames]

    def get_frame(self):
        """Get the most recent frame"""
        with self.lock:
            if self.latest_frame is not None:
                self.processed_count += 1
                return self.latest_frame.copy()
            return None
    
    def get_frame_with_timestamp(self):
        """Get the most recent frame with timestamp"""
        with self.lock:
            if self.latest_frame is not None:
                self.processed_count += 1
                return self.latest_frame.copy(), self.latest_frame_time
            return None, 0

    def stop(self):
        """Stop the frame reader"""
        self.stopped = True
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()