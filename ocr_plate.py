from ultralytics import YOLO
import torch

class OCRPlate:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
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

    def map_class_name(self, class_id):
        # return self.class_id_map.get(class_id, self.province_map.get(class_id, 'Unknown'))
        return self.class_id_map.get(int(self.model.names[class_id]), '')
    def map_province_name(self,class_id):
        return self.province_map.get(int(self.model.names[class_id]), '')

    def predict(self, image):
        if image:
            results = self.model.predict(image, device=self.device,verbose = False)
            main_class_bboxes, province_bboxes = [], []
            
            for bbox in results[0].boxes:
                class_id = int(bbox.cls[0].item())
                if class_id <= 53:
                    main_class_bboxes.append(bbox)
                else:
                    province_bboxes.append(bbox)
            
            sorted_main_bboxes = sorted(main_class_bboxes, key=lambda x: x.xywh[0][0].item())
            sorted_class_names = [self.map_class_name(int(bbox.cls[0].item())) for bbox in sorted_main_bboxes]
            province_class_names = [self.map_province_name(int(bbox.cls[0].item())) for bbox in province_bboxes]
            print(sorted_class_names)
            return ''.join(sorted_class_names) if sorted_class_names else "Unknown", province_class_names[0] if province_class_names else "Unknown"
        else :
            return None, None
