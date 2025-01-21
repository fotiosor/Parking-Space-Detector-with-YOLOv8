# Car_Parking_Space_Detector_YOLOv8




#AI #YOLOv8 #SmartParking #ObjectDetection #DeepLearning #ComputerVision #AIinAction #Innovation #ParkingTech #TechSolutions #Streamlit

## Description
This project is a **Parking Space Detector** leveraging the **YOLOv8** object detection model and **Streamlit** for real-time video analysis. The application processes video streams to:

1. Detect vehicles in a parking lot.
2. Analyze the status of parking spaces (free or occupied).
3. Support detection for both normal and disabled parking spaces.
4. Display real-time information, including timers for how long each space has been occupied.

The application is user-friendly and supports custom parking space configurations loaded via a JSON file.

---

## Features
- **YOLOv8 Object Detection**: Accurately detects vehicles and classifies parking spaces as free or occupied.
- **Normal and Disabled Parking Support**: Differentiates between regular and disabled parking spaces with distinct visual indicators.
- **Real-Time Visualization**: Overlays parking space statuses directly on video frames.
- **Customizable Parking Layouts**: Easily define parking space configurations through JSON files.
- **Interactive Dashboard**: Built with **Streamlit** for an intuitive user experience.
- **Output Video**: Generates a video file with processed frames, including overlays and statistics.

---

## Requirements

### Software Dependencies
- Python 3.8+
- Streamlit
- OpenCV
- PyTorch
- Numpy
- ultralytics

### Hardware Requirements
- A system capable of running Python and handling video processing.
- **GPU (recommended)**: Enhances YOLOv8 inference speed, especially for large videos.

---

## Setup

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**Note**: Ensure the `ultralytics` library is installed to utilize YOLOv8.

### 3. Add YOLOv8 Model
Download the YOLOv8 model weights (e.g., `yolov8n.pt`) and place them in the project directory.

### 4. Prepare JSON File
Create or use an existing JSON file to define parking space configurations. The JSON file format is as follows:
```json
[
    {
        "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "type": 0  // 0 for normal, 1 for disabled
    },
    {
        "points": [[x5, y5], [x6, y6], [x7, y7], [x8, y8]],
        "type": 1
    }
]
```

### 5. Run the Application
Launch the Streamlit application:
```bash
streamlit run appstreamlit.py
```

---

## Usage

### 1. Upload Files
- **Video file** (`.mp4`): Input video for processing.
- **JSON file** (`.json`): Parking space configuration.

### 2. Process Video
- The application analyzes the video frame by frame, detecting vehicles and updating parking statuses. Disabled parking spaces are highlighted distinctly from normal spaces.

### 3. Download Results
- After processing, download the annotated output video as an `.mp4` file.

---

## Code Overview

### Key Components

#### 1. `drawPolygons`
Draws parking spaces on video frames and updates their statuses (free/occupied). Differentiates normal and disabled spaces with distinct colors.

#### 2. `YOLO_Detection`
Utilizes YOLOv8 to detect vehicles in each video frame.

#### 3. `load_positions_from_json`
Loads parking space definitions from a JSON file.

#### 4. `draw_legend`
Displays a legend overlay showing statistics and timers for parking spaces, including separate indicators for normal and disabled spaces.

#### 5. `process_video`
Processes the video in real-time, applying YOLO detection, updating parking statuses, and saving the output.

---

## Example JSON File
```json
[
    {
        "points": [[100, 200], [200, 200], [200, 300], [100, 300]],
        "type": 0
    },
    {
        "points": [[300, 200], [400, 200], [400, 300], [300, 300]],
        "type": 1
    }
]
```

---

## Output
- **Real-Time Visualization**: Annotated video with parking statuses displayed on the Streamlit dashboard. Disabled spaces are shown in a distinct color.
- **Processed Video**: A downloadable `.mp4` file with annotated parking spaces and statistics.

---

## Contributing
Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests to improve the project.

---

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
