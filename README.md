# Product Detection and Grouping AI Pipeline

## Overview
This AI pipeline application provides automated product detection and brand grouping for retail shelf images using computer vision and machine learning techniques.

## Features
- 🖼️ **Product Detection**: YOLOv5-based detection of products in retail shelf images
- 🏷️ **Intelligent Grouping**: Multi-strategy grouping (spatial, visual, brand-based)
- 🎨 **Visualization**: Color-coded bounding boxes with group legends
- 🚀 **Scalable Architecture**: Microservices design with async processing
- 📊 **RESTful API**: Easy integration with other systems
- 🐳 **Docker Support**: Containerized deployment

## Architecture

### Components
1. **Flask Web Server**: Handles HTTP requests and serves web interface
2. **Detection Service**: YOLOv5 model for object detection
3. **Grouping Service**: Multi-strategy product grouping (spatial, visual, brand)
4. **Async Processing Queue**: Handles concurrent requests
5. **Redis Queue**: (Optional) For distributed processing

### Processing Pipeline
