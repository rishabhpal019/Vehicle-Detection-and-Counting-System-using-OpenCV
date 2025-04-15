# Vehicle Detection and Counting System using OpenCV

## Overview

The **Vehicle Detection and Counting System** is a computer vision-based project developed using Python and OpenCV. This system aims to detect moving vehicles from a video feed (either live or recorded) and count them as they pass through a defined detection line. It is suitable for applications such as traffic monitoring, road management, and vehicle flow analysis.

This project uses background subtraction techniques, contour detection, and tracking logic to identify vehicles and count them accurately. The system is lightweight and can be deployed on edge devices with minimal configuration.

## Features

- Real-time vehicle detection from video streams
- Customizable counting line and region of interest (ROI)
- Frame-by-frame tracking of vehicles using contours
- Supports multiple video formats and resolutions
- Displays total vehicle count on-screen
- Easy to modify and extend (e.g., integrate with a database or cloud service)

## How It Works

1. **Input Video Feed**: A video stream is read using OpenCV's `cv2.VideoCapture()`.
2. **Preprocessing**: Frames are resized and converted to grayscale.
3. **Background Subtraction**: A subtractor such as MOG2 is applied to identify moving objects (vehicles).
4. **Contour Detection**: Contours are extracted from the thresholded frame to find the shape of moving vehicles.
5. **Filtering**: Small contours are filtered out based on area to reduce noise.
6. **Counting Line Logic**: A virtual line is placed across the frame. When a vehicle crosses this line, it is counted.
7. **Display & Output**: The live video feed shows vehicle detection in real-time along with the count displayed on the screen.

## Requirements

- Python 3.6+
- OpenCV (cv2)
- Numpy

To install the required packages, you can use:

```bash
pip install -r requirements.txt
