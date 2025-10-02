import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

class ResultVisualizer:
    def __init__(self):
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
    
    def draw_detections(self, image_bytes: bytes, detections: list) -> bytes:
        """Рисует bbox'ы на изображении и возвращает bytes"""
        # Конвертируем bytes в numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PIL для рисования текста
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Пробуем загрузить шрифт, иначе используем стандартный
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Рисуем каждый bbox
        for i, detection in enumerate(detections):
            color = self.colors[i % len(self.colors)]
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Рисуем прямоугольник
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Рисуем подпись с фоном
            label = f"{class_name} {confidence:.2f}"
            bbox_text = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
        
        # Конвертируем обратно в bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()