import threading
import time
import logging
from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

@dataclass
class FramePacket:
    """Container for a frame and its associated metadata"""
    frame: np.ndarray
    event: Dict[str, Any]
    timestamp: float
    priority: int = 5
    processed: bool = False
    
    @property
    def age(self) -> float:
        """Return age of frame in seconds"""
        return time.time() - self.timestamp

class TimeSyncedFrameBuffer:
    """
    Thread-safe buffer for frames with precise timestamp synchronization
    Manages frames and their associated events with priorities
    """
    def __init__(self, max_size=30, timeout=5.0):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.timeout = timeout  # Maximum age of frames before they're considered stale
        
        # Stats for monitoring
        self.frames_added = 0
        self.frames_retrieved = 0
        self.frames_dropped = 0
        self.frames_expired = 0
        self.avg_wait_time = 0
        self.last_stats_time = time.time()
        
        logging.info(f"Frame buffer initialized with max size {max_size}")
    
    def put(self, frame: np.ndarray, event: Dict[str, Any], timestamp: float, priority: int = 5) -> None:
        """Add a frame to the buffer with its metadata"""
        with self.lock:
            # Create packet with all necessary metadata
            packet = FramePacket(
                frame=frame.copy(),  # Copy frame to ensure it's not modified externally
                event=event.copy() if event else {},
                timestamp=timestamp,
                priority=priority
            )
            
            # Add to buffer
            self.buffer.append(packet)
            self.frames_added += 1
            
            # Notify waiting threads
            self.not_empty.notify()
    
    def get(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, Dict[str, Any], float]]:
        """
        Get the highest priority frame that needs processing
        Returns (frame, event, timestamp) tuple or None if buffer is empty
        """
        with self.not_empty:
            start_wait = time.time()
            
            # Wait for data if buffer is empty
            if len(self.buffer) == 0:
                self.not_empty.wait(timeout)
            
            # If still empty after wait, return None
            if len(self.buffer) == 0:
                return None
            
            # Find the highest priority unprocessed packet
            best_packet = None
            best_idx = -1
            best_priority = -1
            
            for idx, packet in enumerate(self.buffer):
                # Skip already processed packets
                if packet.processed:
                    continue
                    
                # Skip expired packets
                if packet.age > self.timeout:
                    packet.processed = True
                    self.frames_expired += 1
                    continue
                    
                # Find highest priority packet
                if packet.priority > best_priority:
                    best_packet = packet
                    best_idx = idx
                    best_priority = packet.priority
            
            # If no suitable packet found
            if best_packet is None:
                # Check if we should clean up buffer
                self._cleanup()
                return None
                
            # Mark as processed
            best_packet.processed = True
            self.frames_retrieved += 1
            
            # Update wait time stats
            wait_time = time.time() - start_wait
            self.avg_wait_time = 0.9 * self.avg_wait_time + 0.1 * wait_time
            
            return (best_packet.frame, best_packet.event, best_packet.timestamp)
    
    def get_frame_for_event_time(self, event_time: float, tolerance: float = 0.2) -> Optional[Tuple[np.ndarray, float]]:
        """
        Find the closest frame to a specific event time
        Returns (frame, actual_timestamp) tuple or None if no suitable frame
        """
        with self.lock:
            if len(self.buffer) == 0:
                return None
                
            # Find the closest frame to event_time
            best_packet = None
            best_time_diff = float('inf')
            
            for packet in self.buffer:
                time_diff = abs(packet.timestamp - event_time)
                if time_diff < best_time_diff and time_diff <= tolerance:
                    best_packet = packet
                    best_time_diff = time_diff
            
            if best_packet:
                return (best_packet.frame.copy(), best_packet.timestamp)
            return None
    
    def _cleanup(self) -> None:
        """Remove expired/processed frames to avoid memory buildup"""
        with self.lock:
            # Count before cleanup
            before_count = len(self.buffer)
            
            # Create new buffer with only unprocessed and non-expired frames
            now = time.time()
            active_packets = [p for p in self.buffer 
                             if not p.processed and (now - p.timestamp) <= self.timeout]
            
            # Update stats
            removed = before_count - len(active_packets)
            if removed > 0:
                self.frames_dropped += removed
                self.buffer = deque(active_packets, maxlen=self.buffer.maxlen)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics"""
        with self.lock:
            total = len(self.buffer)
            unprocessed = sum(1 for p in self.buffer if not p.processed)
            
            return {
                "buffer_size": total,
                "unprocessed": unprocessed,
                "frames_added": self.frames_added,
                "frames_retrieved": self.frames_retrieved,
                "frames_dropped": self.frames_dropped,
                "frames_expired": self.frames_expired,
                "avg_wait_time_ms": self.avg_wait_time * 1000,
                "buffer_usage_percent": (total / self.buffer.maxlen) * 100 if self.buffer.maxlen else 0,
            }
    
    def log_stats(self) -> None:
        """Log buffer statistics"""
        now = time.time()
        if now - self.last_stats_time >= 10:  # Log every 10 seconds
            stats = self.get_stats()
            logging.info(f"Frame buffer stats: {stats['buffer_size']}/{self.buffer.maxlen} frames, "
                        f"{stats['unprocessed']} unprocessed, "
                        f"Added: {stats['frames_added']}, Retrieved: {stats['frames_retrieved']}, "
                        f"Dropped: {stats['frames_dropped']}, "
                        f"Avg wait: {stats['avg_wait_time_ms']:.1f}ms")
            self.last_stats_time = now

    def clear(self) -> None:
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()