Turn any camera into a real-time AI agent in 10 minutes. Tested end-to-end from iPhone 15 capture to Windows 11 RTX 4070 with PyTorch + ONNX + TensorRT FP16 optimization, this project combines YOLOv8 Nano, CUDA, TensorRT, and OpenCV for ultra-low latency object detection.
_____________________________________________________________________________________________________________________________________________________________________________________________
Overview
This project transforms any standard camera into a real-time AI agent capable of object detection in 60 minutes. It leverages the NVIDIA ecosystem for maximum performance using YOLOv8 Nano, PyTorch, Ultralytics, CUDA, ONNX, and TensorRT on a GeForce RTX 4070 with Intel Core i9.
The goal is to demonstrate high-speed, low-latency AI inference on consumer devices while showcasing practical, real-world deployment techniques.
_____________________________________________________________________________________________________________________________________________________________________________________________
Key Features
Real-time video capture and processing
YOLOv8 Nano model trained in PyTorch and exported via ONNX, optimized for FP16 TensorRT inference
Low-latency performance (~8–10ms per frame on RTX 4070)
Cross-device workflow tested with iPhone 15 video capture
Modular code structure for microservices and additional models
_____________________________________________________________________________________________________________________________________________________________________________________________
Tech Stack
Language: Python 3.10+
Deep Learning Frameworks: PyTorch, YOLOv8 Nano (Ultralytics, exported to ONNX)
Acceleration: CUDA, TensorRT (FP16)
Computer Vision: OpenCV for video capture and visualization
Hardware: GeForce RTX 4070, Intel Core i9, NVIDIA G-Sync
OS: Windows 11
_____________________________________________________________________________________________________________________________________________________________________________________________
Installation
Clone the repository:
git clone https://github.com/<jarvisp7>/AI-VisionCamera.git
cd AI-Camera
Create a virtual environment and install dependencies:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Ensure your system has the CUDA Toolkit and TensorRT installed compatible with your RTX 4070.
_____________________________________________________________________________________________________________________________________________________________________________________________
Usage
Run the main script:
python ai_camera.py
Workflow tested:
Capture video on iPhone 15 camera
AirDrop the video to a MacBook
Attempted to send via Gmail, but the file was too large
Stored the video in Google Drive for accessibility
Access the video from a Windows 11 laptop with GeForce RTX 4070, Intel Core i9, and NVIDIA G-Sync
Run the script to process the video with YOLOv8 Nano trained in PyTorch, exported to ONNX, and optimized with TensorRT FP16
Output is displayed in real-time using OpenCV with bounding boxes
This workflow demonstrates cross-device integration, handling large media files, cloud storage management, and GPU-accelerated inference
Use FP16 mode for best performance
_____________________________________________________________________________________________________________________________________________________________________________________________
Architecture
[Video Capture (iPhone 15)] → [Transfer: MacBook / Google Drive] → [Preprocessing] → [YOLOv8 Nano PyTorch → ONNX → TensorRT FP16 Engine] → [Postprocessing] → [Display (Windows 11)]
Video Capture: iPhone 15 records video for input
Transfer: Video moved via AirDrop and Google Drive to Windows 11 laptop
Preprocessing: Resizes frames, normalizes, prepares tensors
YOLOv8 Nano PyTorch → ONNX → TensorRT FP16 Engine: Optimized inference for real-time detection
Postprocessing: Parses detections and applies bounding boxes
Display: OpenCV window visualizes results live
_____________________________________________________________________________________________________________________________________________________________________________________________
Performance Metrics
Frame Rate: ~120 FPS at 640x640 input resolution
Latency: ~8–10ms per frame (TensorRT FP16)
Cross-device workflow verified: iPhone 15 → MacBook → Google Drive → Windows 11 + RTX 4070
GPU Utilization: Efficient usage of RTX 4070
_____________________________________________________________________________________________________________________________________________________________________________________________
Roadmap
Complete NVIDIA NIM microservices integration
Expand PPE detection and additional real-time models
Add cross-platform support for Linux and Mac
_____________________________________________________________________________________________________________________________________________________________________________________________
References
PyTorch: https://pytorch.org/
Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
ONNX: https://onnx.ai/
NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
OpenCV: https://opencv.org/
_____________________________________________________________________________________________________________________________________________________________________________________________
Author
Jarvis Perdue– U.S. Army Veteran, Sr. HVAC Engineer, Senior Device Management Engineer, Sr. Audio Engineer and Music Producer
Focused on practical AI applications, real-time deployment, and NVIDIA ecosystem innovation
