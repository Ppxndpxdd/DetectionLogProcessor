import queue
import threading
import logging
import time
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import cv2
import heapq

@dataclass
class FramePacket:
    """Container for frame data and metadata with quality metrics"""
    frame: np.ndarray
    event: Dict[str, Any]
    timestamp: float
    frame_id: int
    quality: float = 0.0
    processing_priority: int = 0

class TimeSyncedFrameBuffer:
    """
    Advanced frame buffer with precise time synchronization capabilities
    """
    def __init__(self, max_size=30):
        # Main processing queue (priority queue)
        self.processing_queue = queue.PriorityQueue(maxsize=max_size)
        
        # Time-indexed frame storage for precise matching
        self.time_indexed_frames = {}  # timestamp -> frame
        self.timestamps = []  # Sorted timestamps for binary search
        
        # Locks
        self.lock = threading.RLock()
        self.index_lock = threading.RLock()
        
        # Stats
        self.frame_counter = 0
        self.dropped_frames = 0
        self.processed_frames = 0
        self.last_stats_time = time.time()
        self.sync_quality = deque(maxlen=100)  # Track time sync accuracy

    def add_frame(self, frame: np.ndarray, timestamp: float, quality: float = 0.5):
        """Add a frame to the time-indexed buffer"""
        if frame is None or not isinstance(frame, np.ndarray):
            return False
            
        with self.index_lock:
            # Store frame with timestamp
            self.time_indexed_frames[timestamp] = frame.copy()
            
            # Keep timestamps sorted for efficient search
            bisect_pos = self._find_insertion_point(timestamp)
            self.timestamps.insert(bisect_pos, timestamp)
            
            # Trim buffer if needed
            if len(self.timestamps) > 100:  # Keep up to 100 frames in the index
                old_ts = self.timestamps.pop(0)
                del self.time_indexed_frames[old_ts]
                
        return True
        
    def _find_insertion_point(self, timestamp):
        """Find insertion point in sorted timestamps list (binary search)"""
        lo, hi = 0, len(self.timestamps)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.timestamps[mid] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo
        
    def get_frame_at_time(self, target_time, max_diff_ms=100):
        """
        Get frame closest to the specified time using binary search
        Very efficient time-based frame lookup
        """
        with self.index_lock:
            if not self.timestamps:
                return None, 0
                
            # Binary search for closest timestamp
            idx = self._find_closest_timestamp_index(target_time)
            if idx is None:
                return None, 0
                
            closest_time = self.timestamps[idx]
            time_diff_ms = abs(closest_time - target_time) * 1000
            
            if time_diff_ms <= max_diff_ms:
                self.sync_quality.append(time_diff_ms / max_diff_ms)
                return self.time_indexed_frames[closest_time].copy(), closest_time
            else:
                self.sync_quality.append(1.0)  # Poor sync quality
                return None, 0
                
    def _find_closest_timestamp_index(self, target_time):
        """Find index of timestamp closest to target using binary search"""
        if not self.timestamps:
            return None
            
        # Fast path for common cases
        if target_time <= self.timestamps[0]:
            return 0
        if target_time >= self.timestamps[-1]:
            return len(self.timestamps) - 1
            
        # Binary search
        lo, hi = 0, len(self.timestamps) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.timestamps[mid] == target_time:
                return mid
            elif self.timestamps[mid] < target_time:
                lo = mid + 1
            else:
                hi = mid - 1
                
        # At this point, lo > hi and self.timestamps[hi] < target_time < self.timestamps[lo]
        # Return the closer timestamp
        if lo >= len(self.timestamps):
            return hi
        if hi < 0:
            return lo
            
        if target_time - self.timestamps[hi] < self.timestamps[lo] - target_time:
            return hi
        return lo

    def put(self, frame: np.ndarray, event: Dict[str, Any], timestamp: float = None) -> bool:
        """Add a frame to the processing queue with priority"""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.frame_counter += 1
            frame_id = self.frame_counter
            
            # Determine priority based on event type and timing
            event_type = event.get('event', '').lower()
            priority = 5  # Default priority
            
            # Vehicle events get higher priority
            if 'vehicle' in event_type or 'car' in event_type:
                priority -= 2
            elif 'person' in event_type:
                priority -= 1
                
            # Newer events get higher priority
            frame_age = time.time() - timestamp
            if frame_age < 0.1:  # Very fresh frames
                priority -= 2
            elif frame_age < 0.3:  # Fresh frames
                priority -= 1
            elif frame_age > 1.0:  # Old frames
                priority += 2
            
            # Bound priority
            priority = max(0, min(9, priority))
            
            # Create frame packet
            packet = FramePacket(
                frame=frame.copy(),
                event=event.copy(),
                timestamp=timestamp,
                frame_id=frame_id,
                quality=0.5,  # Default quality
                processing_priority=priority
            )
            
            # Add to queue, drop lowest priority if full
            try:
                self.processing_queue.put_nowait((priority, packet))
                return True
            except queue.Full:
                try:
                    # Drop lowest priority item
                    self.processing_queue.get_nowait()
                    self.processing_queue.task_done()
                    self.processing_queue.put_nowait((priority, packet))
                    self.dropped_frames += 1
                    return True
                except (queue.Empty, queue.Full):
                    self.dropped_frames += 1
                    return False

    def get(self, timeout=1.0) -> Optional[FramePacket]:
        """Get next frame packet from buffer based on priority"""
        try:
            priority, packet = self.processing_queue.get(timeout=timeout)
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
        with self.lock:
            queue_size = self.processing_queue.qsize()
            
            avg_sync_quality = 0
            if self.sync_quality:
                avg_sync_quality = 1.0 - (sum(self.sync_quality) / len(self.sync_quality))
                
            logging.info(f"Frame Buffer: Queue={queue_size}, "
                         f"Processed={self.processed_frames}, Dropped={self.dropped_frames}, "
                         f"Time Index Size={len(self.timestamps)}, "
                         f"Sync Quality={avg_sync_quality:.2f}")

    def task_done(self):
        """Mark task as done"""
        self.processing_queue.task_done()

    def clear(self):
        """Clear all items from buffers"""
        with self.lock:
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.task_done()
                except queue.Empty:
                    break
                
        with self.index_lock:
            self.time_indexed_frames.clear()
            self.timestamps.clear()