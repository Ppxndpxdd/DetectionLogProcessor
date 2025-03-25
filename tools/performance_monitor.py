import threading
import time
import logging
import json
import os
from collections import deque
import psutil
import numpy as np
from typing import Dict, Any, List

class PerformanceMonitor:
    """
    Real-time system performance monitor with adaptive tuning capabilities
    """
    def __init__(self, config_path=None, update_interval=5.0):
        self.update_interval = update_interval
        self.running = True
        self.stats = {
            'cpu_usage': 0,
            'memory_usage_mb': 0,
            'processing_fps': 0,
            'event_latency_ms': 0,
            'ocr_latency_ms': 0,
            'detection_latency_ms': 0,
            'buffer_usage': 0,
            'event_backlog': 0,
            'system_load': 0
        }
        self.processing_times = deque(maxlen=100)
        self.ocr_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.event_latencies = deque(maxlen=100)
        self.fps_history = deque(maxlen=10)
        self.lock = threading.RLock()
        
        # Components to monitor
        self.frame_buffer = None
        self.ocr_pool = None
        self.plate_detector = None
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
            
        # Start monitoring thread
        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="Performance-Monitor"
        )
        self.thread.start()
        
    def register_components(self, frame_buffer=None, ocr_pool=None, plate_detector=None):
        """Register system components to monitor"""
        self.frame_buffer = frame_buffer
        self.ocr_pool = ocr_pool
        self.plate_detector = plate_detector
    
    def update_processing_time(self, time_value):
        """Record a processing time measurement"""
        with self.lock:
            self.processing_times.append(time_value)
            
    def update_ocr_time(self, time_value):
        """Record an OCR processing time measurement"""
        with self.lock:
            self.ocr_times.append(time_value)
            
    def update_detection_time(self, time_value):
        """Record a detection time measurement"""
        with self.lock:
            self.detection_times.append(time_value)
            
    def update_event_latency(self, time_value):
        """Record event-to-output latency"""
        with self.lock:
            self.event_latencies.append(time_value)
            
    def get_stats(self):
        """Get current performance statistics"""
        with self.lock:
            return self.stats.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_time = time.time()
        last_frames_processed = 0
        frames_processed = 0
        
        while self.running:
            try:
                # Sleep for update interval
                time.sleep(self.update_interval)
                current_time = time.time()
                elapsed = current_time - last_time
                
                # Get system metrics
                process = psutil.Process()
                cpu_percent = process.cpu_percent() / psutil.cpu_count()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Calculate FPS
                if hasattr(self.frame_buffer, 'get_stats'):
                    buffer_stats = self.frame_buffer.get_stats()
                    current_frames = buffer_stats.get('frames_retrieved', 0)
                    new_frames = current_frames - last_frames_processed
                    fps = new_frames / elapsed if elapsed > 0 else 0
                    self.fps_history.append(fps)
                    avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                    last_frames_processed = current_frames
                else:
                    buffer_stats = {}
                    avg_fps = 0
                
                # Calculate latencies
                with self.lock:
                    avg_processing = np.mean(self.processing_times) * 1000 if self.processing_times else 0
                    avg_ocr = np.mean(self.ocr_times) * 1000 if self.ocr_times else 0
                    avg_detection = np.mean(self.detection_times) * 1000 if self.detection_times else 0
                    avg_latency = np.mean(self.event_latencies) * 1000 if self.event_latencies else 0
                
                # Get OCR pool stats
                ocr_stats = self.ocr_pool.get_stats() if self.ocr_pool else {}
                
                # Calculate system load (0-100%)
                buffer_usage = buffer_stats.get('buffer_usage_percent', 0)
                ocr_queue = ocr_stats.get('queue_size', 0)
                event_backlog = ocr_stats.get('queue_size', 0) + buffer_stats.get('unprocessed', 0)
                system_load = min(100, (cpu_percent + buffer_usage/2 + ocr_queue*10) / 2)
                
                # Update stats
                with self.lock:
                    self.stats.update({
                        'cpu_usage': cpu_percent,
                        'memory_usage_mb': memory_mb,
                        'processing_fps': avg_fps,
                        'event_latency_ms': avg_latency,
                        'ocr_latency_ms': avg_ocr,
                        'detection_latency_ms': avg_detection,
                        'buffer_usage': buffer_usage,
                        'event_backlog': event_backlog,
                        'system_load': system_load,
                        'timestamp': current_time
                    })
                
                # Log performance stats
                logging.info(f"Performance: CPU={cpu_percent:.1f}%, "
                            f"Mem={memory_mb:.1f}MB, "
                            f"FPS={avg_fps:.1f}, "
                            f"Latency={avg_latency:.1f}ms, "
                            f"OCR={avg_ocr:.1f}ms, "
                            f"Load={system_load:.1f}%")
                
                # Adaptive tuning based on system load
                self._tune_system_parameters(system_load, event_backlog)
                
                last_time = current_time
                
            except Exception as e:
                logging.error(f"Error in performance monitor: {e}", exc_info=True)
                
    def _tune_system_parameters(self, system_load, event_backlog):
        """Dynamically tune system parameters based on load"""
        try:
            # Adjust OCR confidence threshold
            if self.ocr_pool and hasattr(self.ocr_pool.ocr_instance, 'set_backlog'):
                self.ocr_pool.ocr_instance.set_backlog(event_backlog)
                
            # Adjust plate detector parameters    
            if self.plate_detector and hasattr(self.plate_detector, 'set_backlog'):
                self.plate_detector.set_backlog(event_backlog)
                
            # Under extreme load, drop low-priority tasks
            if system_load > 90:
                logging.warning(f"System under extreme load ({system_load:.1f}%), activating emergency mode")
                # Implementation depends on specific system
        except Exception as e:
            logging.error(f"Error tuning system parameters: {e}")
    
    def shutdown(self):
        """Shutdown the monitoring thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)