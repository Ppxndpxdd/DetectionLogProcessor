import queue
import threading
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import deque

@dataclass
class FramePacket:
    """Container for frame data and metadata"""
    frame: np.ndarray
    event: Dict[str, Any]
    timestamp: float
    frame_id: int

class FrameBuffer:
    """
    Thread-safe buffer for frames with overwrite policy
    """
    def __init__(self, max_size=30):
        self.buffer = queue.Queue(maxsize=max_size)
        self.recent_frames = deque(maxlen=10)  # Store most recent frames with timestamps
        self.lock = threading.RLock()
        self.frame_counter = 0
        self.dropped_frames = 0
        self.processed_frames = 0
        self.last_stats_time = time.time()
        
    def add_recent_frame(self, frame, timestamp):
        """Add frame to recent frames buffer"""
        if frame is not None and isinstance(frame, np.ndarray):
            with self.lock:
                self.recent_frames.append((frame.copy(), timestamp))
    
    def get_best_frame_for_event(self, event_time, max_age=0.5):
        """Find the best frame matching the event time"""
        best_frame = None
        best_time_diff = float('inf')
        best_timestamp = 0
        
        with self.lock:
            for frame, timestamp in self.recent_frames:
                time_diff = abs(timestamp - event_time)
                # Choose the closest frame that's not from the future
                if time_diff < best_time_diff and timestamp <= event_time + 0.05:
                    best_frame = frame
                    best_time_diff = time_diff
                    best_timestamp = timestamp
        
        if best_frame is not None and best_time_diff < max_age:
            return best_frame.copy(), best_timestamp
        return None, 0
    
    def put(self, frame: np.ndarray, event: Dict[str, Any], timestamp: float = None) -> bool:
        """
        Add a frame to the buffer, dropping oldest if full
        Returns True if successful, False if dropped
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.frame_counter += 1
            frame_id = self.frame_counter
            
            # Create frame packet with metadata
            packet = FramePacket(
                frame=frame.copy(),  # Ensure we have a copy
                event=event.copy() if event else {},  # Copy event dict
                timestamp=timestamp,
                frame_id=frame_id
            )
            
            # Try to add to buffer, drop oldest if full
            try:
                self.buffer.put_nowait(packet)
                return True
            except queue.Full:
                # Drop oldest frame to make room
                try:
                    _ = self.buffer.get_nowait()
                    self.buffer.task_done()
                    self.buffer.put_nowait(packet)
                    self.dropped_frames += 1
                    return True
                except (queue.Empty, queue.Full):
                    self.dropped_frames += 1
                    return False
    
    def get(self, timeout=1.0) -> Optional[FramePacket]:
        """Get next frame packet from buffer"""
        try:
            packet = self.buffer.get(timeout=timeout)
            self.processed_frames += 1
            
            # Log stats periodically
            current_time = time.time()
            if current_time - self.last_stats_time >= 30:
                self.log_stats()
                self.last_stats_time = current_time
                
            return packet
        except queue.Empty:
            return None
    
    def log_stats(self):
        """Log buffer statistics"""
        logging.info(f"Frame Buffer Stats: Queue Size={self.buffer.qsize()}, "
                     f"Recent Frames={len(self.recent_frames)}, "
                     f"Processed={self.processed_frames}, Dropped={self.dropped_frames}")
    
    def task_done(self):
        """Mark task as done"""
        self.buffer.task_done()
    
    def clear(self):
        """Clear all items from buffer"""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                    self.buffer.task_done()
                except queue.Empty:
                    break
            self.recent_frames.clear()