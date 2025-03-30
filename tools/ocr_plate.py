from ultralytics import YOLO
import torch
import time
import logging
import numpy as np
from PIL import Image
import cv2
from functools import lru_cache
from threading import RLock

class OCRPlate:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        
        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
            logging.info("mps device selected")
        elif torch.cuda.is_available():
            device = "cuda"
            logging.info("cuda device selected")
        else:
            device = "cpu"
            logging.warning("No GPU available, falling back to CPU")
        
        self.device = device
        self.class_id_map = {i: str(i) for i in range(10)}
        self.class_id_map.update({
            10: 'ก', 11: 'ข', 12: 'ฃ', 13: 'ค', 14: 'ฅ', 15: 'ฆ', 16: 'ง', 17: 'จ', 18: 'ฉ', 19: 'ช',
            20: 'ซ', 21: 'ฌ', 22: 'ญ', 23: 'ฎ', 24: 'ฏ', 25: 'ฐ', 26: 'ฑ', 27: 'ฒ', 28: 'ณ', 29: 'ด',
            30: 'ต', 31: 'ถ', 32: 'ท', 33: 'ธ', 34: 'น', 35: 'บ', 36: 'ป', 37: 'ผ', 38: 'ฝ', 39: 'พ',
            40: 'ฟ', 41: 'ภ', 42: 'ม', 43: 'ย', 44: 'ร', 45: 'ล', 46: 'ว', 47: 'ศ', 48: 'ษ', 49: 'ส',
            50: 'ห', 51: 'ฬ', 52: 'อ', 53: 'ฮ'
        })
        
        self.province_map = {i: province for i, province in enumerate([
            'กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท',
            'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม',
            'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์',
            'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่',
            'พะเยา', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยะลา', 'ยโสธร', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง',
            'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ',
            'สมุทรสงคราม', 'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย',
            'หนองบัวลำภู', 'อ่างทอง', 'อุดรธานี', 'อุทัยธานี', 'อุตรดิตถ์', 'อุบลราชธานี', 'อำนาจเจริญ'
        ], start=54)}
        
        # Performance tracking
        self.total_reads = 0
        self.successful_reads = 0
        self.ocr_times = []
        self.cache_hits = 0
        self.cache_lock = RLock()
        self.recent_plates = {}  # Cache recently seen plates by object_id
        
        # Adaptive parameters
        self.confidence_threshold = 0.1  # Start with moderate confidence
        self.backlog = 0  # Track processing backlog
        
        logging.info(f"OCR Plate initialized on {self.device} device")

    def map_class_name(self, class_id):
        """Safely map class ID to character, handling potential errors"""
        try:
            # Convert class_id to int to ensure proper lookup
            class_id_int = int(class_id)
            
            # First try to get class name from model.names
            model_class_name = self.model.names.get(class_id_int)
            
            # If model_class_name is valid, use it to lookup in class_id_map
            if model_class_name is not None:
                return self.class_id_map.get(int(model_class_name), '')
            
            # If not found in model.names, try direct lookup in class_id_map
            return self.class_id_map.get(class_id_int, '')
        except Exception as e:
            logging.warning(f"Error mapping class name for ID {class_id}: {e}")
            return ''
        
    def map_province_name(self, class_id):
        """Safely map class ID to province name, handling potential errors"""
        try:
            # Convert class_id to int to ensure proper lookup
            class_id_int = int(class_id)
            
            # First try to get class name from model.names
            model_class_name = self.model.names.get(class_id_int)
            
            # If model_class_name is valid, use it to lookup in province_map
            if model_class_name is not None:
                return self.province_map.get(int(model_class_name), 'Unknown')
            
            # If not found in model.names, try direct lookup in province_map
            return self.province_map.get(class_id_int, 'Unknown')
        except Exception as e:
            logging.warning(f"Error mapping province name for ID {class_id}: {e}")
            return 'Unknown'
    
    def predict(self, image, object_id=None):
        """Predict license plate number and province with caching and optimization"""
        # Check for None or empty image
        if image is None:
            return None, None
        
        # Check dimensions for numpy arrays
        if isinstance(image, np.ndarray):
            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                return None, None
            
        start_time = time.time()
        self.total_reads += 1
        
        try:
            # Check cache for this object if ID provided
            if object_id and object_id in self.recent_plates:
                with self.cache_lock:
                    cached_data = self.recent_plates[object_id]
                    # Check if cache entry is fresh (less than 10 seconds old)
                    if time.time() - cached_data['timestamp'] < 10.0:
                        plate_number = cached_data.get('plate_number')
                        province = cached_data.get('province')
                        
                        # Validate cached data is complete and valid
                        if plate_number and province and plate_number != "Unknown":
                            self.cache_hits += 1
                            logging.info(f"Cache hit for object {object_id}: {plate_number} {province}")
                            return plate_number, province
                        else:
                            logging.warning(f"Invalid cache data for object {object_id}, running prediction")
            
            # REMOVED: Preprocessing step - using raw image directly
            # This speeds up processing by eliminating expensive image transformations
            
            # Ensure image is in correct format for the model
            input_image = image
            if isinstance(image, np.ndarray) and image.ndim == 3:
                # Convert BGR to RGB if needed
                if image.shape[2] == 3:
                    # Check if the image needs color conversion (OpenCV uses BGR by default)
                    if not isinstance(image, Image.Image):
                        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run prediction with optimized settings
            results = self.model.predict(
                input_image, 
                device=self.device, 
                verbose=False,
                conf=0.1,
                half=True
            )
            
            # Process results with safe tensor handling and character validation
            if len(results) > 0 and results[0].boxes.data.shape[0] > 0:
                main_class_bboxes = []
                province_bboxes = []
                
                # Extract and categorize detections with safe tensor handling
                for i in range(results[0].boxes.data.shape[0]):
                    try:
                        # Safe extraction with .item() to convert tensor to Python scalar
                        cls_tensor = results[0].boxes.cls[i]
                        class_id = int(cls_tensor.item() if hasattr(cls_tensor, 'item') else int(cls_tensor))
                        
                        # Store detection with box
                        if class_id <= 53:  # Characters
                            main_class_bboxes.append((i, class_id))
                        else:  # Province
                            province_bboxes.append((i, class_id))
                    except Exception as e:
                        logging.warning(f"Error processing detection {i}: {e}")
                        continue
                
                # Sort characters by x-position
                sorted_main_bboxes = []
                if main_class_bboxes:
                    # Safe extraction and sorting of x positions
                    positions = []
                    for idx, class_id in main_class_bboxes:
                        try:
                            # Safely extract x coordinate
                            x_coord = results[0].boxes.xywh[idx][0]
                            x_value = float(x_coord.item() if hasattr(x_coord, 'item') else x_coord)
                            positions.append((idx, class_id, x_value))
                        except Exception as e:
                            logging.warning(f"Error extracting position for {idx}: {e}")
                    
                    # Sort by x position
                    sorted_positions = sorted(positions, key=lambda p: p[2])
                    sorted_main_bboxes = [(idx, class_id) for idx, class_id, _ in sorted_positions]
                
                # Enhanced character validation for Thai characters
                sorted_class_names = []
                valid_char_count = 0
                for _, class_id in sorted_main_bboxes:
                    char = self.map_class_name(class_id)
                    if char:  # Only add non-empty characters
                        sorted_class_names.append(char)
                        valid_char_count += 1
                
                # Require at least 2 valid characters for a plate number
                if valid_char_count >= 2:
                    plate_number = ''.join(sorted_class_names)
                else:
                    plate_number = "Unknown"
                
                # Get province with validation
                province = "Unknown"
                if province_bboxes:
                    _, province_class_id = province_bboxes[0]
                    province = self.map_province_name(province_class_id)
                    # Verify province is a valid string
                    if not isinstance(province, str) or not province:
                        province = "Unknown"
                
                # Cache result if object_id provided
                if object_id and plate_number != "Unknown" and province != "Unknown":
                    with self.cache_lock:
                        self.recent_plates[object_id] = {
                            'plate_number': plate_number,
                            'province': province if province else "Unknown",  # Ensure province is never None
                            'timestamp': time.time()
                        }
                        
                        # Maintain cache size
                        if len(self.recent_plates) > 100:
                            oldest_key = min(self.recent_plates.keys(), 
                                           key=lambda k: self.recent_plates[k]['timestamp'])
                            del self.recent_plates[oldest_key]
                
                # Ensure return values are never None
                self.successful_reads += 1
                self.ocr_times.append(time.time() - start_time)
                return plate_number if plate_number else None, province if province else "Unknown"
            else:
                return None, None
                
        except Exception as e:
            logging.error(f"Error in OCR prediction: {e}", exc_info=True)
            return None, None
            
    def get_stats(self):
        """Return performance statistics"""
        if not self.ocr_times:
            return {
                "avg_time": 0, 
                "success_rate": 0, 
                "device": self.device, 
                "cache_hit_rate": 0
            }
            
        stats = {
            "avg_time": sum(self.ocr_times[-50:]) / max(len(self.ocr_times[-50:]), 1),
            "success_rate": self.successful_reads / max(self.total_reads, 1) * 100,
            "total_reads": self.total_reads,
            "successful_reads": self.successful_reads,
            "device": self.device,
            "confidence": self.confidence_threshold,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_reads, 1) * 100,
            "backlog": self.backlog
        }
        return stats