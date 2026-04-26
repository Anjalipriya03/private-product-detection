import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self):
        logger.info("Initializing detection service with OpenCV")
        
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def detect_products(self, image):
       
        detections = []
        
        edge_detections = self._detect_by_edges(image)
        detections.extend(edge_detections)
        
        color_detections = self._detect_by_color(image)
        detections.extend(color_detections)
        
        hog_detections = self._detect_by_hog(image)
        detections.extend(hog_detections)
        
        detections = self._non_maximum_suppression(detections)
        
        logger.info(f"Total detected objects: {len(detections)}")
        return detections
    
    def _detect_by_edges(self, image):
        detections = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        min_area = (w * h) * 0.005  
        max_area = (w * h) * 0.5    
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                if 0.3 < aspect_ratio < 3:
                    # Calculate confidence based on edge density
                    edge_density = np.sum(edges[y:y+h_box, x:x+w_box]) / (w_box * h_box * 255)
                    confidence = min(0.95, 0.5 + edge_density * 0.5)
                    
                    detection = {
                        'bbox': [x, y, x + w_box, y + h_box],
                        'confidence': float(confidence),
                        'class_id': 0,
                        'class_name': 'product',
                        'center': [x + w_box // 2, y + h_box // 2],
                        'area': float(area)
                    }
                    detections.append(detection)
        
        return detections
    
    def _detect_by_color(self, image):
        detections = []
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_ranges = [
            ([0, 50, 50], [10, 255, 255]),    # Red
            ([10, 50, 50], [25, 255, 255]),   # Orange
            ([25, 50, 50], [35, 255, 255]),   # Yellow
            ([35, 50, 50], [85, 255, 255]),   # Green
            ([85, 50, 50], [125, 255, 255]),  # Blue
            ([125, 50, 50], [145, 255, 255]), # Purple
            ([145, 50, 50], [180, 255, 255])  # Pink
        ]
        
        h, w = image.shape[:2]
        min_area = (w * h) * 0.005
        
        for lower, upper in color_ranges:
            # Create mask for color range
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect_ratio = w_box / h_box if h_box > 0 else 0
                    
                    if 0.3 < aspect_ratio < 3:
                        detection = {
                            'bbox': [x, y, x + w_box, y + h_box],
                            'confidence': 0.6,  # Moderate confidence for color-based
                            'class_id': 0,
                            'class_name': 'product',
                            'center': [x + w_box // 2, y + h_box // 2],
                            'area': float(area)
                        }
                        detections.append(detection)
        
        return detections
    
    def _detect_by_hog(self, image):
        detections = []
        
        # Resize image for faster processing
        h, w = image.shape[:2]
        scale = min(640 / w, 480 / h) if max(w, h) > 640 else 1
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image
            new_w, new_h = w, h
        
        # Detect people/objects using HOG
        rects, weights = self.hog.detectMultiScale(resized, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        # Scale back to original image size
        if scale < 1:
            rects = [(int(x / scale), int(y / scale), int(w_box / scale), int(h_box / scale)) 
                    for (x, y, w_box, h_box) in rects]
        
        for (x, y, w_box, h_box), weight in zip(rects, weights):
            if weight > 0.5:  # Confidence threshold
                detection = {
                    'bbox': [x, y, x + w_box, y + h_box],
                    'confidence': float(weight),
                    'class_id': 0,
                    'class_name': 'product',
                    'center': [x + w_box // 2, y + h_box // 2],
                    'area': float(w_box * h_box)
                }
                detections.append(detection)
        
        return detections
    
    def _non_maximum_suppression(self, detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []
        
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        picked = []
        boxes = [d['bbox'] for d in detections]
        
        while len(boxes) > 0:
            current = boxes[0]
            picked.append(detections[0])
            
            if len(boxes) == 1:
                break
            
            remaining = []
            remaining_detections = []
            
            for i in range(1, len(boxes)):
                # Calculate IoU
                box = boxes[i]
                intersection_x1 = max(current[0], box[0])
                intersection_y1 = max(current[1], box[1])
                intersection_x2 = min(current[2], box[2])
                intersection_y2 = min(current[3], box[3])
                
                intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
                
                current_area = (current[2] - current[0] + 1) * (current[3] - current[1] + 1)
                box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                
                iou = intersection_area / float(current_area + box_area - intersection_area)
                
                if iou < iou_threshold:
                    remaining.append(box)
                    remaining_detections.append(detections[i])
            
            boxes = remaining
            detections = remaining_detections
        
        return picked