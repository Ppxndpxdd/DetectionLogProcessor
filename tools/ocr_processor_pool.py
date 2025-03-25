import threading
import queue
import time
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional, Callable

class OCRProcessorPool:
    """
    Dedicated thread pool for OCR processing with priority queue
    and load balancing to handle the 200ms OCR bottleneck
    """
    
    def __init__(self, num_workers=2, ocr_instance=None):
        self.ocr_instance = ocr_instance
        self.input_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.cache_lock = threading.RLock()
        self.num_workers = num_workers
        self.worker_threads = []
        self.shutdown_flag = threading.Event()
        self.stats = {
            'processed': 0,
            'queued': 0,
            'cache_hits': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'max_queue_size': 0,
        }
        self.stats_lock = threading.RLock()
        
        # Start worker threads
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker_thread,
                daemon=True,
                name=f"OCR-Pool-Worker-{i+1}"
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logging.info(f"OCR Processor Pool initialized with {num_workers} workers")
    
    def submit(self, plate_img, object_id, callback: Callable, priority=5):
        """
        Submit a plate image for OCR processing
        
        Args:
            plate_img: The cropped plate image
            object_id: Object ID for tracking
            callback: Function to call with results (plate_number, province, object_id)
            priority: Processing priority (lower number = higher priority)
        """
        if plate_img is None:
            return False
            
        # Check if we already have this object_id in processing or results
        with self.cache_lock:
            if object_id in self.result_cache:
                cached_result = self.result_cache[object_id]
                if time.time() - cached_result['timestamp'] < 5.0:  # Use cache if less than 5 seconds old
                    plate_number = cached_result.get('plate_number')
                    province = cached_result.get('province')
                    
                    # FIX: Validate cached results before using
                    if plate_number and province and plate_number != "Unknown":
                        with self.stats_lock:
                            self.stats['cache_hits'] += 1
                        
                        # Call callback with cached results
                        callback(plate_number, province, object_id)
                        return True
                    else:
                        logging.info(f"Invalid cached result for object {object_id}, running new OCR")
        
        # Queue for processing
        queue_item = (priority, time.time(), {
            'plate_img': plate_img,
            'object_id': object_id,
            'callback': callback,
            'queued_time': time.time()
        })
        
        self.input_queue.put(queue_item)
        
        with self.stats_lock:
            self.stats['queued'] += 1
            current_queue_size = self.input_queue.qsize()
            if current_queue_size > self.stats['max_queue_size']:
                self.stats['max_queue_size'] = current_queue_size
                
        return True
    
    def _worker_thread(self):
        """Worker thread to process OCR tasks"""
        logging.info(f"Starting OCR worker thread: {threading.current_thread().name}")
        
        while not self.shutdown_flag.is_set():
            try:
                # Get next item with timeout to allow checking shutdown flag
                try:
                    priority, submit_time, task = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                # Extract task data
                plate_img = task['plate_img']
                object_id = task['object_id']
                callback = task['callback']
                queued_time = task['queued_time']
                
                # Skip if too old (over 5 seconds in queue)
                if time.time() - queued_time > 5.0:
                    logging.warning(f"Skipping stale OCR task for object {object_id}, "
                                   f"queued for {time.time() - queued_time:.1f}s")
                    self.input_queue.task_done()
                    continue
                
                # Process OCR
                start_time = time.time()
                plate_number, province = self.ocr_instance.predict(plate_img, object_id)
                processing_time = time.time() - start_time
                
                # Cache result
                if plate_number and province:
                    with self.cache_lock:
                        self.result_cache[object_id] = {
                            'plate_number': plate_number,
                            'province': province,
                            'timestamp': time.time()
                        }
                        
                        # Limit cache size
                        if len(self.result_cache) > 200:
                            oldest_key = min(self.result_cache.keys(), 
                                           key=lambda k: self.result_cache[k]['timestamp'])
                            del self.result_cache[oldest_key]
                
                # Call callback with results
                try:
                    callback(plate_number, province, object_id)
                except Exception as e:
                    logging.error(f"Error in OCR callback: {e}")
                
                # Update statistics
                with self.stats_lock:
                    self.stats['processed'] += 1
                    self.stats['total_processing_time'] += processing_time
                    self.stats['avg_processing_time'] = (
                        self.stats['total_processing_time'] / self.stats['processed']
                    )
                    
                # Log processing time
                if processing_time > 0.3:  # Log slow processing
                    logging.info(f"OCR processing took {processing_time*1000:.1f}ms for object {object_id}")
                
                # Mark task as done
                self.input_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error in OCR worker: {e}\n{traceback.format_exc()}")
                time.sleep(0.1)  # Short sleep on error to prevent CPU spike
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.stats_lock:
            stats_copy = self.stats.copy()
            stats_copy['queue_size'] = self.input_queue.qsize()
            stats_copy['worker_count'] = self.num_workers
            stats_copy['cache_size'] = len(self.result_cache)
            return stats_copy
    
    def shutdown(self):
        """Shutdown the processor pool"""
        self.shutdown_flag.set()
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        logging.info("OCR Processor Pool shut down")