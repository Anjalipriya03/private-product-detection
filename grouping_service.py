import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import colorsys
import random

class GroupingService:
    def __init__(self, eps=50, min_samples=2, similarity_threshold=0.7):
        """
        Initialize grouping service
        
        Args:
            eps: Maximum distance between samples for DBSCAN clustering
            min_samples: Minimum number of samples in a neighborhood
            similarity_threshold: Threshold for feature-based grouping
        """
        self.eps = eps
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
        
        # Brand mapping (in production, would use ML model)
        self.brand_patterns = {
            'huggies': ['huggies', 'diaper', 'baby', 'huggies_logo'],
            'head_shoulders': ['head & shoulders', 'shampoo', 'hair', 'dandruff'],
            'coca_cola': ['coca', 'cola', 'coke', 'soda'],
            'pepsi': ['pepsi', 'soda', 'cola'],
            'lays': ['lays', 'chips', 'snack', 'potato'],
            'doritos': ['doritos', 'chips', 'tortilla'],
            'oreo': ['oreo', 'cookie', 'biscuit'],
            'tide': ['tide', 'detergent', 'laundry'],
            'dove': ['dove', 'soap', 'body wash']
        }
        
    def extract_features(self, image, bbox):
        """Extract visual features from detected product region"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Extract color histogram features
        hist_features = self._extract_color_histogram(roi)
        
        # Extract texture features (HOG descriptor)
        hog_features = self._extract_hog_features(roi)
        
        # Extract edge density
        edge_features = self._extract_edge_density(roi)
        
        # Combine features
        features = np.concatenate([hist_features, hog_features, edge_features])
        
        return features
    
    def _extract_color_histogram(self, roi, bins=32):
        """Extract color histogram features"""
        hist_features = []
        
        for channel in range(3):
            hist = cv2.calcHist([roi], [channel], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def _extract_hog_features(self, roi):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        try:
            # Resize to consistent size
            roi_resized = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Simple HOG implementation using Sobel
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            
            # Quantize angles into bins
            bin_n = 9
            bins = np.int32(bin_n * ang / (360 + 0.001))
            bin_cells = bins[:32, :32], bins[32:, :32], bins[:32, 32:], bins[32:, 32:]
            mag_cells = mag[:32, :32], mag[32:, :32], mag[:32, 32:], mag[32:, 32:]
            
            hog_features = []
            for bin_cell, mag_cell in zip(bin_cells, mag_cells):
                hist = np.bincount(bin_cell.ravel(), mag_cell.ravel(), minlength=bin_n)
                hog_features.extend(hist)
            
            return np.array(hog_features)
        except:
            return np.zeros(36)  # Return zero features if extraction fails
    
    def _extract_edge_density(self, roi):
        """Extract edge density features"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        return np.array([edge_density])
    
    def group_by_spatial_proximity(self, detections, image_shape):
        """Group products based on spatial proximity"""
        if len(detections) < 2:
            return [0] * len(detections)
        
        # Extract centers
        centers = np.array([d['center'] for d in detections])
        
        # Normalize coordinates
        scaler = StandardScaler()
        centers_scaled = scaler.fit_transform(centers)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps / 100, min_samples=self.min_samples)
        labels = clustering.fit_predict(centers_scaled)
        
        return labels
    
    def group_by_visual_similarity(self, image, detections):
        """Group products based on visual similarity"""
        if len(detections) < 2:
            return [0] * len(detections)
        
        features = []
        for detection in detections:
            feat = self.extract_features(image, detection['bbox'])
            if feat is not None:
                features.append(feat)
            else:
                features.append(np.zeros(100))
        
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        similarities = np.dot(features_scaled, features_scaled.T)
        
        labels = np.zeros(len(detections), dtype=int)
        current_label = 0
        
        for i in range(len(detections)):
            if labels[i] == 0:
                current_label += 1
                labels[i] = current_label
                
                for j in range(i + 1, len(detections)):
                    if labels[j] == 0 and similarities[i, j] > self.similarity_threshold:
                        labels[j] = current_label
        
        return labels - 1 
    
    def assign_brand_groups(self, detections):
        group_ids = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            brand_found = False
            
            for brand_id, patterns in self.brand_patterns.items():
                for pattern in patterns:
                    if pattern in class_name:
                        group_ids.append(brand_id)
                        brand_found = True
                        break
                if brand_found:
                    break
            
            if not brand_found:
                # Assign generic group based on class
                group_ids.append(f"generic_{detection['class_id']}")
        
        unique_groups = {}
        numeric_ids = []
        next_id = 0
        
        for group in group_ids:
            if group not in unique_groups:
                unique_groups[group] = next_id
                next_id += 1
            numeric_ids.append(unique_groups[group])
        
        return numeric_ids, unique_groups
    
    def group_products(self, detections, image):
        
        if len(detections) == 0:
            return []
        
        
        spatial_labels = self.group_by_spatial_proximity(detections, image.shape)
        
        
        visual_labels = self.group_by_visual_similarity(image, detections)
        
        
        brand_labels, brand_mapping = self.assign_brand_groups(detections)
        
        combined_labels = []
        for i, detection in enumerate(detections):
            if brand_labels[i] != 0:
                group_id = f"brand_{brand_labels[i]}"
            elif visual_labels[i] != -1:
                group_id = f"visual_{visual_labels[i]}"
            else:
                group_id = f"spatial_{spatial_labels[i]}"
            combined_labels.append(group_id)
        
        grouped_detections = []
        unique_groups = {}
        
        for i, detection in enumerate(detections):
            group_id = combined_labels[i]
            
            if group_id not in unique_groups:
                unique_groups[group_id] = len(unique_groups) + 1
            
            grouped_detection = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'group_id': unique_groups[group_id],
                'group_name': group_id,
                'center': detection['center']
            }
            grouped_detections.append(grouped_detection)
        
        return grouped_detections
    
    def visualize_results(self, image, grouped_detections, output_path):
        
        vis_image = image.copy()
        
        # Generate distinct colors for each group
        num_groups = max([d['group_id'] for d in grouped_detections]) if grouped_detections else 0
        colors = {}
        
        for i in range(1, num_groups + 1):
            hue = i / num_groups if num_groups > 0 else 0.5
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colors[i] = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        # Draw bounding boxes and labels
        for detection in grouped_detections:
            x1, y1, x2, y2 = detection['bbox']
            group_id = detection['group_id']
            color = colors[group_id]
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"Group {group_id}: {detection['class_name']}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend
        legend_x = 10
        legend_y = 30
        legend_spacing = 25
        
        cv2.putText(vis_image, "Group Legend:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for group_id, color in colors.items():
            y = legend_y + legend_spacing * group_id
            cv2.rectangle(vis_image, (legend_x, y - 15), (legend_x + 20, y), color, -1)
            cv2.putText(vis_image, f"Group {group_id}", (legend_x + 25, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        
        return output_path