import queue
import threading
import logging
import time
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import cv2

@dataclass
class FramePacket:
    """Container for frame data and metadata with quality metrics"""
    frame: np.ndarray
    event: Dict[str, Any]
    timestamp: float
    frame_id: int
    quality: float = 0.0
    processing_priority: int = 0

class EventPrediction:
    """Predicts and tracks object position based on movement patterns"""
    def __init__(self, event_history_length=5):
        self.object_history = {}  # object_id -> list of (timestamp, bbox) tuples
        self.max_history = event_history_length
    
    def update(self, object_id, timestamp, bbox):
        """Update object position history"""
        if object_id not in self.object_history:
            self.object_history[object_id] = []
            
        history = self.object_history[object_id]
        history.append((timestamp, bbox))
        
        # Trim history
        if len(history) > self.max_history:
            history.pop(0)
    
    def predict_position(self, object_id, target_time):
        """Predict object position at the given time based on movement history"""
        history = self.object_history.get(object_id, [])
        if len(history) < 2:
            return None
            
        # Simple linear prediction based on last two positions
        (t1, bbox1), (t2, bbox2) = history[-2], history[-1]
        
        if t1 == t2:
            return bbox2  # No movement, return last known position
            
        # Calculate velocity for each bbox parameter
        time_diff = t2 - t1
        velocity = [(b2 - b1) / time_diff for b1, b2 in zip(bbox1, bbox2)]
        
        # Predict position at target_time
        time_to_target = target_time - t2
        predicted_bbox = [b2 + v * time_to_target for b2, v in zip(bbox2, velocity)]
        
        # Ensure normalized coordinates stay in range [0,1]
        return [max(0, min(1, p)) for p in predicted_bbox]

class FrameBuffer:
    """
    High-performance buffer with predictive frame matching and quality assessment
    """
    def __init__(self, max_size=30):
        self.processing_queue = queue.PriorityQueue(maxsize=max_size)  # Priority queue for processing
        self.recent_frames = deque(maxlen=90)  # Store frames with timestamps (90 = ~3s at 30fps)
        self.lock = threading.RLock()
        self.frame_counter = 0
        self.dropped_frames = 0
        self.processed_frames = 0
        self.last_stats_time = time.time()
        
        # Performance optimization
        self.downsample_factor = 0.25  # For quick frame comparison
        self.last_frame_gray = None
        
        # Event prediction
        self.event_predictor = EventPrediction()
        
        # Performance tracking
        self.processing_latency = deque(maxlen=100)
        self.matching_accuracy = deque(maxlen=100)
    
    def _calculate_frame_quality(self, frame):
        """Calculate quality metrics for a frame"""
        try:
            if frame is None:
                return 0.0
                
            # Downsample for faster processing
            small_gray = cv2.cvtColor(
                cv2.resize(frame, (0, 0), fx=self.downsample_factor, fy=self.downsample_factor),
                cv2.COLOR_BGR2GRAY
            )
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(small_gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast
            contrast = small_gray.std()
            
            # Calculate motion if we have a previous frame
            motion_score = 0
            if self.last_frame_gray is not None:
                # Calculate frame difference
                if small_gray.shape == self.last_frame_gray.shape:
                    diff = cv2.absdiff(small_gray, self.last_frame_gray)
                    motion_score = np.mean(diff)
            
            self.last_frame_gray = small_gray
            
            # Combined quality score (0-1)
            quality = (
                min(1.0, sharpness / 2000) * 0.4 +  # Sharpness component
                min(1.0, contrast / 80) * 0.3 +     # Contrast component
                min(1.0, motion_score / 30) * 0.3    # Motion component
            )
            
            return quality
            
        except Exception as e:
            logging.error(f"Error calculating frame quality: {e}")
            return 0.5  # Default mid-range quality
    
    def add_recent_frame(self, frame: np.ndarray, timestamp: float):
        """Add a frame to the recent frames buffer with quality assessment"""
        if frame is not None and isinstance(frame, np.ndarray):
            with self.lock:
                # Calculate quality
                quality = self._calculate_frame_quality(frame)
                
                # Store frame with quality info
                self.recent_frames.append((frame.copy(), timestamp, quality))
    
    def get_best_frame_for_event(self, event):
        """
        Find the best frame matching the event using advanced heuristics:
        1. Time-based matching
        2. Position prediction for moving objects
        3. Quality-based selection
        """
        with self.lock:
            if not self.recent_frames:
                return None, 0
                
            event_time = event.get('first_seen')
            if isinstance(event_time, str):
                try:
                    event_time = float(event_time)
                except ValueError:
                    event_time = time.time()
            elif event_time is None:
                event_time = time.time()
            
            object_id = event.get('object_id', 'unknown')
            bbox = event.get('bbox')
            
            # Update event predictor if we have bbox info
            if bbox and len(bbox) == 4:
                self.event_predictor.update(object_id, event_time, bbox)
            
            # Criteria weights for frame selection
            TIME_WEIGHT = 0.6
            QUALITY_WEIGHT = 0.4
            
            best_frame = None
            best_score = -float('inf')
            best_timestamp = 0
            
            # First pass: find frames with timestamps close to event_time
            candidates = []
            for frame, timestamp, quality in self.recent_frames:
                time_diff = abs(timestamp - event_time)
                
                # Only consider frames within 0.5 seconds of event time
                if time_diff <= 0.5:
                    time_score = 1.0 - min(1.0, time_diff * 2.0)  # 0-1.0 based on time difference
                    score = time_score * TIME_WEIGHT + quality * QUALITY_WEIGHT
                    candidates.append((frame, timestamp, quality, score))
            
            # If we found candidates, pick the best one
            if candidates:
                # Sort by score (highest first)
                candidates.sort(key=lambda x: x[3], reverse=True)
                best_frame, best_timestamp, _, best_score = candidates[0]
                
                matching_quality = min(1.0, 1.0 - abs(best_timestamp - event_time))
                self.matching_accuracy.append(matching_quality)
                
                return best_frame.copy(), best_timestamp
            
            # Fall back to closest frame if no good candidates
            closest_frame = None
            closest_time = 0
            min_diff = float('inf')
            
            for frame, timestamp, _ in self.recent_frames:
                diff = abs(timestamp - event_time)
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = frame
                    closest_time = timestamp
            
            if closest_frame is not None:
                # Track poor match quality
                self.matching_accuracy.append(0.0)
                return closest_frame.copy(), closest_time
            
            # No frames available
            return None, 0
    
    def put(self, frame: np.ndarray, event: Dict[str, Any], timestamp: float = None) -> bool:
        """
        Add a frame to the processing queue with priority based on quality and event type
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.frame_counter += 1
            frame_id = self.frame_counter
            
            # Calculate frame quality
            quality = self._calculate_frame_quality(frame)
            
            # Determine processing priority (lower = higher priority)
            # Base priority on event type and quality
            priority = 5  # Default priority
            
            event_type = event.get('event', '').lower()
            if 'vehicle' in event_type or 'car' in event_type:
                priority -= 2  # Higher priority for vehicles
            elif 'person' in event_type:
                priority -= 1  # Medium priority for people
                
            # Better quality frames get higher priority
            priority -= int(quality * 2)
            
            # Bound priority between 0-9
            priority = max(0, min(9, priority))
            
            # Create frame packet with metadata
            packet = FramePacket(
                frame=frame.copy(),
                event=event.copy(),
                timestamp=timestamp,
                frame_id=frame_id,
                quality=quality,
                processing_priority=priority
            )
            
            # Try to add to buffer, drop lowest priority if full
            try:
                # For PriorityQueue, we need a tuple with priority first
                self.processing_queue.put_nowait((priority, packet))
                return True
            except queue.Full:
                # Buffer is full, try to get one item to make space
                try:
                    self.processing_queue.get_nowait()  # This removes lowest priority item
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
            start_time = time.time()
            priority, packet = self.processing_queue.get(timeout=timeout)
            processing_latency = time.time() - start_time
            self.processing_latency.append(processing_latency)
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
        """Log detailed buffer statistics"""
        with self.lock:
            queue_size = self.processing_queue.qsize()
            recent_frame_count = len(self.recent_frames)
            
            avg_latency = np.mean(self.processing_latency) if self.processing_latency else 0
            avg_matching = np.mean(self.matching_accuracy) if self.matching_accuracy else 0
            
            logging.info(
                f"Frame Buffer Stats: Queue={queue_size}/{self.processing_queue.maxsize}, "
                f"Recent Frames={recent_frame_count}/{self.recent_frames.maxlen}, "
                f"Processed={self.processed_frames}, Dropped={self.dropped_frames}, "
                f"Avg Latency={avg_latency*1000:.1f}ms, Matching Quality={avg_matching:.2f}"
            )
    
    def task_done(self):
        """Mark task as done"""
        self.processing_queue.task_done()
    
    def clear(self):
        """Clear all items from buffer"""
        with self.lock:
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.task_done()
                except queue.Empty:
                    break
            self.recent_frames.clear()