import cv2
import numpy as np
import requests
import os
import time

def create_varied_test_images():
    """Create multiple test images with different product arrangements"""
    
    images = []
    
    # Image 1: Clustered products (should group together)
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    for i in range(5):
        x = 50 + i * 80
        cv2.rectangle(img1, (x, 100), (x+60, 250), (0, 0, 255), -1)
        cv2.putText(img1, f"Product{i+1}", (x+5, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite('test_clustered.jpg', img1)
    images.append('test_clustered.jpg')
    
    # Image 2: Scattered products (different groups)
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    positions = [(50, 100), (300, 150), (500, 80), (100, 300), (450, 250)]
    for i, (x, y) in enumerate(positions):
        cv2.rectangle(img2, (x, y), (x+80, y+100), (255, 0, 0), -1)
        cv2.putText(img2, f"Item{i+1}", (x+5, y+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite('test_scattered.jpg', img2)
    images.append('test_scattered.jpg')
    
    # Image 3: Similar products (same brand/type)
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    colors = [(0, 0, 255)] * 4  # All red boxes
    for i in range(4):
        x = 50 + i * 120
        cv2.rectangle(img3, (x, 150), (x+80, 280), colors[i], -1)
        cv2.putText(img3, "Coca-Cola", (x+5, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite('test_similar.jpg', img3)
    images.append('test_similar.jpg')
    
    print(f"Created {len(images)} test images:")
    for img in images:
        print(f"  - {img}")
    
    return images

def test_api_with_image(image_path, api_url="http://localhost:5000"):
    """Test the API with an image"""
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Upload image
    with open(image_path, 'rb') as f:
        files = {'image': f}
        print(f"\nUploading {image_path}...")
        response = requests.post(f"{api_url}/upload", files=files)
    
    if response.status_code != 202:
        print(f"Upload failed: {response.status_code}")
        return None
    
    data = response.json()
    request_id = data['request_id']
    print(f"Request ID: {request_id}")
    
    # Poll for results
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(1)
        status_response = requests.get(f"{api_url}/status/{request_id}")
        status_data = status_response.json()
        
        if status_data['status'] == 'completed':
            print(f"Processing completed!")
            print(f"Total objects detected: {status_data.get('total_objects', 0)}")
            print(f"Unique groups: {status_data.get('unique_groups', 0)}")
            
            # Get full results
            result_response = requests.get(f"{api_url}/result/{request_id}")
            result_data = result_response.json()
            
            # Save visualization
            vis_response = requests.get(f"{api_url}/visualization/{request_id}")
            if vis_response.status_code == 200:
                vis_filename = f"vis_{os.path.basename(image_path)}"
                with open(vis_filename, 'wb') as f:
                    f.write(vis_response.content)
                print(f"Visualization saved: {vis_filename}")
            
            return result_data
            
        elif status_data['status'] == 'failed':
            print(f"Processing failed: {status_data.get('error', 'Unknown error')}")
            return None
    
    print("Timeout waiting for processing")
    return None

if __name__ == "__main__":
    # Create test images
    print("Creating test images...")
    test_images = create_varied_test_images()
    
    # Test each image
    for image_path in test_images:
        result = test_api_with_image(image_path)
        if result:
            print(f"Successfully processed {image_path}")
            print(f"Found {result.get('total_objects', 0)} products in {result.get('unique_groups', 0)} groups")
        print("-" * 50)