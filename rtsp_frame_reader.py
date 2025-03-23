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
    High-performance RTSP frame reader with hardware acceleration 
    and adaptive frame management for real-time video processing.
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
        self.quality_threshold = 0.5  # Dynamic quality threshold
        self.skip_frames_count = 3    # Dynamic frame skip count
        
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
            # transport = "udp" if self.rtsp_url.startswith("rtsp://") else "tcp"
            transport = "tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|"
                "buffer_size;16384|"
                "max_delay;0|"
                "stimeout;2000000|"
                "reorder_queue_size;0"
            )
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            # self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Configure for minimal latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            
            # Try hardware acceleration if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logging.info("CUDA hardware acceleration available")
                # CUDA settings would go here
            
            logging.info("Using FFMPEG backend for RTSP")
        
        if self.cap.isOpened():
            logging.info(f"RTSP stream opened: {self.rtsp_url}")
            
            # Adjust capture parameters
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logging.info(f"Stream properties: {width}x{height} @ {fps}fps")
        else:
            logging.error(f"Failed to open RTSP stream: {self.rtsp_url}")

    def _calculate_frame_quality(self, frame):
        """Calculate frame quality metrics for adaptive processing"""
        if frame is None:
            return 0
            
        try:
            # Fast quality assessment using downsampled grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            
            # Sharpness - Laplacian variance
            laplacian = cv2.Laplacian(small, cv2.CV_64F)
            self.frame_sharpness = laplacian.var()
            
            # Brightness
            self.frame_brightness = np.mean(small)
            
            # Motion detection if we have previous frame
            if self.prev_frame is not None:
                diff = cv2.absdiff(small, self.prev_frame)
                self.motion_score = np.mean(diff)
                
            self.prev_frame = small
                
            # Combined quality score (0-1)
            if self.frame_sharpness > 500:  # Empirical threshold
                quality = min(1.0, self.frame_sharpness / 2000.0) * 0.7 + self.motion_score * 0.3
            else:
                quality = 0.3 + self.motion_score * 0.7
                
            return quality
            
        except Exception as e:
            logging.error(f"Error calculating frame quality: {e}")
            return 0.5  # Default quality

    def run(self):
        """Main acquisition thread - continuously captures frames with adaptive timing"""
        last_frame_time = 0
        last_log_time = time.time()
        frames_since_log = 0
        consecutive_empty_frames = 0
        
        while not self.stopped:
            try:
                # Adaptive frame skipping based on system performance
                for _ in range(max(1, self.skip_frames_count)):
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
                    
                    # Calculate quality and decide whether to use this frame
                    frame_quality = self._calculate_frame_quality(frame)
                    
                    # Keep high-quality frames, or if we haven't had a good one in a while
                    if frame_quality >= self.quality_threshold or consecutive_empty_frames >= 10:
                        # Store frame with timestamp and quality info
                        with self.lock:
                            self.latest_frame = frame
                            self.latest_frame_time = current_time
                            self.frame_deque.append((frame.copy(), current_time, frame_quality))
                        consecutive_empty_frames = 0
                    else:
                        self.skipped_count += 1
                        consecutive_empty_frames += 1
                    
                    # Update adaptive parameters based on system performance
                    if len(self.frame_interval_history) >= 30:
                        mean_interval = np.mean(self.frame_interval_history)
                        
                        # If we're processing frames quickly, be more selective
                        if mean_interval < 0.02:  # < 20ms between frames
                            self.quality_threshold = min(0.7, self.quality_threshold + 0.02)
                            self.skip_frames_count = min(5, self.skip_frames_count + 1)
                        # If we're processing slowly, be less selective
                        elif mean_interval > 0.05:  # > 50ms between frames
                            self.quality_threshold = max(0.3, self.quality_threshold - 0.02)
                            self.skip_frames_count = max(1, self.skip_frames_count - 1)
                    
                    # Log performance metrics every 5 seconds
                    if current_time - last_log_time > 5:
                        elapsed = current_time - last_log_time
                        if elapsed > 0:
                            fps = frames_since_log / elapsed
                            self.fps = fps
                            logging.debug(f"RTSP: {fps:.1f}fps, Quality: {frame_quality:.2f}, "
                                          f"Threshold: {self.quality_threshold:.2f}, "
                                          f"Skip: {self.skip_frames_count}, "
                                          f"Total: {self.frame_count}, Process: {self.processed_count}, "
                                          f"Skip: {self.skipped_count}, Buffer: {len(self.frame_deque)}")
                            frames_since_log = 0
                            last_log_time = current_time
                    
                    last_frame_time = current_time
                else:
                    consecutive_empty_frames += 1
                    
                    if consecutive_empty_frames > 20:
                        logging.warning(f"Multiple empty frames ({consecutive_empty_frames}), reconnecting...")
                        self._reconnect()
                        consecutive_empty_frames = 0
                
                # Sleep adaptively based on frame rate
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
    
    def get_frame_closest_to_time(self, target_time, max_diff=0.5):
        """Get the frame closest to the specified timestamp"""
        with self.lock:
            if not self.frame_deque:
                return None, 0
            
            # Find frame with closest timestamp
            closest_frame = None
            closest_time = 0
            min_diff = float('inf')
            
            for frame, timestamp, quality in self.frame_deque:
                diff = abs(timestamp - target_time)
                if diff < min_diff and diff <= max_diff:
                    min_diff = diff
                    closest_frame = frame
                    closest_time = timestamp
            
            if closest_frame is not None:
                self.processed_count += 1
                return closest_frame.copy(), closest_time
                
            # Fall back to most recent frame if no close match
            best_frame, best_time, _ = self.frame_deque[-1]
            return best_frame.copy(), best_time

    def stop(self):
        """Stop the frame reader"""
        self.stopped = True
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()